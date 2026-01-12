"""
CudaText AI Agents plugin using ACP (Agent Client Protocol)
Multi-agent support - allows multiple concurrent agent sessions


https://agentclientprotocol.com/protocol/overview
https://github.com/agentclientprotocol/agent-client-protocol

Agent Client Protocol SDK (Python). unfortunatly it uses pydantic which we cannot support in cudatext after pydantic v2 because it is compiled now so we have to include a version for each supported python...
https://agentclientprotocol.github.io/python-sdk/
https://github.com/agentclientprotocol/python-sdk/blob/main/examples/gemini.py
https://github.com/agentclientprotocol/python-sdk/blob/main/examples/client.py
"""

# https://github.com/agentclientprotocol/agent-client-protocol/blob/main/schema/meta.json
# https://github.com/agentclientprotocol/agent-client-protocol/blob/main/schema/meta.unstable.json

'''
Summary of cleanup hierarchy:
    Disconnect (_on_disconnect_click) - gui 'Disconnet' button - Graceful, we can reconnect again
    Kill (_on_kill_click) - gui 'Kill' button - Forceful, we can reconnect again
    Hide (_on_dialog_close) - gui 'Close' button - Hide window, agent stays running
    Full cleanup (_on_close_clean_click) - gui 'Close and Clean' button - Kill and remove everything related to the running agent
    Close all (plugin_close_all_agents) - menu command - Remove all panels and all kill all agents
    Exit (on_exit) - Plugin shutdown - Remove all panels and all kill agents

helper functions:
    - _reset_all_ui_and_state(self) - Comprehensive reset of ALL UI elements and state.
        Called when: switching agents, disconnect, kill, dispose, panel close
    - dispose() - kill the agent and clean up resources gracefully
    - kill_agent() - same as dispose but forcefully kill the agent process
    - _cleanup_state() - Clean up client state - called by dispose() and kill_agent() and when process die unexpectedly
'''

'''
cudatext and threads important notes:
=====================================
cudatext freezes when we call some cudatext api's from inside a thread, but there is some exceptions, this api calls do not freeze cudatext:
    msg_status
https://github.com/Alexey-T/CudaText/issues/6176

so for the rest of the api call them from the main thread in on_timer
_________________________________________

fix Access violation error in cudatext:
=======================================
some times i get this errors
ERROR: Exception in CudaText for _chat_timer: Access violation
ERROR: Exception in CudaText for cuda_ai_agents._chat_timer: Access violation
ERROR: Exception in CudaText for on_app_deactivate: Access violation
ERROR: Exception in CudaText for cuda_lsp.on_app_deactivate: Access violation

as you see the error happens in this plugin, and it happens also in lsp client plugin when it fires on_app_deactivate event which use timer_proc(TIMER_STOP.... which fires with every window unfocus
and also my function _chat_timer use timer_proc which fires every 100ms

this error happens a lot of times when i do
time.sleep(0.3);os.startfile('F:\\')
the error does not happen when i use
time.sleep(0.3);subprocess.Popen(['explorer', 'F:\\'])
so the problem is in os.startfile

this is how i understand this error which seems logic but not sound, so i think this is what happen:

os.startfile run in the MAIN THREAD and try to open the file manager window, this seems to take some time to finish, which blocks cudatext's UI, then the timer_proc try to run, then he find that the main thread is bloqued or the time cannot access UI because main thread is blocked, so we get Access violation!

The error happens to LSP too because both timers try to fire while the main thread is blocked. 

solution:
=========
use subprocess.Popen or run os.startfile inside a thread
threading.Thread(target=open_dir, daemon=True).start()

conclusion:
===========
Rule of thumb for CudaText plugins:
- NEVER do in main thread:
time.sleep()
Long-running operations
Network calls
File I/O (if it might be slow)
External process launches
Heavy computations
- ALWAYS do it in worker thread

- Safe in main thread:
CudaText API calls (must be in main thread!)
Quick operations
Queue operations
Spawning worker threads
'''

import os
import sys
import json
import subprocess
import threading
import shutil
import time
from typing import Optional, Callable, Dict, Any, List
from queue import Queue, Empty
from cudatext import *
from cudax_lib import get_translation
import cudatext_cmd as cmds

_ = get_translation(__file__)


status_texts = {
    'disconnected': _('Disconnected'),
    'connecting': _('Connecting...'),
    'connected': _('Connected'),
    'error': _('Error')
}

status_icons = {
    'disconnected': '◯',
    'connecting': '···',
    'connected': '●',
    'error': '✘',
}

status_colors = {
    'disconnected': 0x808080,
    'connecting': 0x0080FF,
    'connected': 0x00C000,
    'error': 0x0000FF,
}

# =============================================================================
# Configuration Management
# =============================================================================

# Configuration file path
CONFIG_FILE = os.path.join(app_path(APP_DIR_SETTINGS), 'cuda_ai_agents_settings.json')
USER_CHOICES_FILE = os.path.join(app_path(APP_DIR_SETTINGS), 'cuda_ai_agents_last_user_choices.json')

def str_to_bool(val):
    """Convert CudaText '0'/'1' string values to Python booleans"""
    # checkbox val returns a string not an int!
    return str(val) == '1'
    
def load_ui_config():
    """Load UI configuration (spy panel, stderr panel visibility)"""
    config = {}
    config_exists = os.path.exists(CONFIG_FILE)
    
    if config_exists:
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            debug_print(f"Failed to load config file: {e}")
    
    # Check if UI settings exist
    has_spy_setting = 'show_spy_panel' in config
    has_stderr_setting = 'show_stderr_panel' in config
    
    # Set defaults if missing
    if not has_spy_setting:
        config['show_spy_panel'] = False
    if not has_stderr_setting:
        config['show_stderr_panel'] = False
    
    # Save config if it was created or updated
    if not config_exists or not has_spy_setting or not has_stderr_setting:
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            debug_print("Saved/updated UI config with defaults")
        except Exception as e:
            debug_print(f"Failed to save UI config: {e}")
    
    return {
        'show_spy_panel': config.get('show_spy_panel', False),
        'show_stderr_panel': config.get('show_stderr_panel', False)
    }

def save_ui_config(show_spy_panel=None, show_stderr_panel=None):
    """Save UI configuration"""
    # Load existing config
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except:
            pass
    
    # Update values
    if show_spy_panel is not None:
        config['show_spy_panel'] = show_spy_panel
    if show_stderr_panel is not None:
        config['show_stderr_panel'] = show_stderr_panel
    
    # Save
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        debug_print("Saved UI config")
    except Exception as e:
        debug_print(f"Failed to save UI config: {e}")
        
def load_and_merge_config():
    """
    Load config file and merge with default AGENTS list.
    User config takes priority, but missing entries are added from defaults.
    Returns: (merged_agents_list, config_dict)
    """
    # Default configuration structure
    config = {
        'servers_id_with_allowed_childs': ['kimi', 'mistral-vibe'],
        'agent_servers': {}
    }
    
    # Load existing config file if it exists
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                # Merge loaded config
                if 'servers_id_with_allowed_childs' in loaded_config:
                    config['servers_id_with_allowed_childs'] = loaded_config['servers_id_with_allowed_childs']
                if 'agent_servers' in loaded_config:
                    config['agent_servers'] = loaded_config['agent_servers']
                debug_print("Loaded config from file")
        except Exception as e:
            debug_print(f"Failed to load config file: {e}")
    
    # Build merged agents list
    merged_agents = []
    config_needs_update = False
    
    # Process each default agent
    for default_agent in AGENTS:
        agent_id = default_agent['id']
        
        if agent_id in config['agent_servers']:
            # User has this agent in config - use config values
            user_agent = config['agent_servers'][agent_id].copy()
            
            # Add any missing keys from default (for new fields in updates)
            for key, value in default_agent.items():
                if key not in user_agent:
                    user_agent[key] = value
                    config_needs_update = True
            
            merged_agents.append(user_agent)
        else:
            # Agent not in config - add default to config
            config['agent_servers'][agent_id] = default_agent.copy()
            merged_agents.append(default_agent.copy())
            config_needs_update = True
    
    # Add user-defined agents that aren't in defaults
    for agent_id, user_agent in config['agent_servers'].items():
        if not any(a['id'] == agent_id for a in AGENTS):
            # This is a user-added agent
            merged_agents.append(user_agent)
    
    # Save config if it was created or updated
    if not os.path.exists(CONFIG_FILE) or config_needs_update:
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2)
            debug_print("Saved/updated config file")
        except Exception as e:
            debug_print(f"Failed to save config file: {e}")
    
    return merged_agents, config

def get_agents_info(return_first_available=False):
    """Get agents with status or return first available agent"""
    # Reload config each time to pick up user changes
    merged_agents, _ = load_and_merge_config()
    
    agents_with_status = []
    first_available = None
    
    for agent in merged_agents:
        agent_copy = agent.copy()
        # Ensure env exists
        if 'env' not in agent_copy:
            agent_copy['env'] = {}
        agent_copy['available'] = EXEC_CACHE.get(agent['command']) is not None
        agents_with_status.append(agent_copy)
        
        if return_first_available and first_available is None and agent_copy['available']:
            first_available = agent_copy
    
    if return_first_available:
        return first_available or (merged_agents[0] if merged_agents else AGENTS[0])
    return agents_with_status

def get_config_settings():
    """Get current config settings (for servers_id_with_allowed_childs, etc.)"""
    _, config = load_and_merge_config()
    return config
    

def load_user_choices():
    """Load user's last choices from file"""
    if not os.path.exists(USER_CHOICES_FILE):
        return {
            'last_agent_id': None,
            'yolo_enabled': False,
            'agent_settings': {}  # agent_id -> {model, mode, predefined_model}
        }
    
    try:
        with open(USER_CHOICES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        debug_print(f"Failed to load user choices: {e}")
        return {
            'last_agent_id': None,
            'yolo_enabled': False,
            'agent_settings': {}
        }

def save_user_choices(choices):
    """Save user's choices to file"""
    try:
        with open(USER_CHOICES_FILE, 'w', encoding='utf-8') as f:
            json.dump(choices, f, indent=2)
        debug_print("Saved user choices")
    except Exception as e:
        debug_print(f"Failed to save user choices: {e}")

def update_user_choice(agent_id, setting_type, value):
    """Update a specific user choice and save immediately"""
    choices = load_user_choices()
    
    if setting_type == 'last_agent_id':
        choices['last_agent_id'] = agent_id
    elif setting_type == 'yolo_enabled':
        choices['yolo_enabled'] = value
    else:
        # Agent-specific settings (model, mode, predefined_model)
        if agent_id not in choices['agent_settings']:
            choices['agent_settings'][agent_id] = {}
        choices['agent_settings'][agent_id][setting_type] = value
    
    save_user_choices(choices)

# =============================================================================
# Windows Job Object Process Creation
# =============================================================================

# Global list to prevent garbage collection of job handles
# If job handles are GC'd, Windows closes them and kills all processes
# in the job (due to JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE flag)
# _ACTIVE_JOB_HANDLES = []

# TODO: add this function to cuda lsp, some lsp servers like python keep running a lot of times after cudatext close 
def create_process_with_cleanup(cmd, cwd=None, allow_childs_of_child=False, env=None):
    """subprocess.Popen wrapper. Create a process with automatic child cleanup on parent death.

    Args:
        - cmd: List of command arguments [executable, arg1, arg2, ...]
        - cwd: Working directory (optional)
        - allow_childs_of_child: (bool) If True, uses JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK.
            - False (Default): The child AND all its descendants are locked in our Job Object. 
            - True: The direct child is in our Job, but its children are free to be assigned 
              to new Job Objects. This is required for tools like 'uv' to work. 
              Cleanup still works because if we kill the direct child ('uv'), its own 
              internal cleanup logic (or its own Job Object) will kill its children.
                               
            WHY USE allow_childs_of_child?
            Some tools (like uv) try to manage their own subprocesses using their own Windows Job Objects.
            
            If we force those subprocesses into OUR Job Object, the tool will fail with:
            "(os error 5) Access is denied" when it tries to assign them to its own job.
            this is not a bug in this function but a bug in uv itself, it had to use CREATE_BREAKAWAY_FROM_JOB flag
        - env: (dict) Environment variables to pass to process. If None, inherits current environment.
             If provided, will be merged with current environment (provided env takes priority).
    
    Returns:
        subprocess.Popen object with automatic cleanup on parent crash
    """
    
    """ why?
    PROBLEM: When CudaText crashes, agent processes become zombies
    ============================================================
    - on_exit() or atexit handlers don't run on crash/force-kill
    - subprocess.Popen has no "kill children when parent dies" mechanism
    - Child processes continue running forever (i saw this with gemini if user lose conexion/is disconnected and attempt to connect to gemini CLI then if cuda crashes gemini keep running)
    
    SOLUTION: Windows Job Objects and Linux PDEATHSIG
    =================================================
    A Job Object is a Windows kernel feature that groups processes together.
    The JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE flag tells Windows: "When the last handle to this job is closed, kill all processes in it"
    
    In Linux things are simpler, we can use os.setpgrp() or os.setpgid()
    When we spawn a child process, it inherits the Process Group ID (PGID) of the parent.
    in popen we use preexec_fn=os.setpgrp this sets the child's PGID to its own PID
    Warning: preexec_fn is considered slightly unsafe in multi-threaded programs because it runs between fork and exec. Modern Python prefers the process_group parameter in subprocess.Popen (added in Python 3.11).
    so a better and simpler way is to use the PR_SET_PDEATHSIG trick, prctl tell the kernel: "Send me a signal if my parent dies."

    ATTEMPTED SOLUTIONS (that failed on windows):
    =============================================
    1. atexit.register() + taskkill.exe
       - Failed: atexit doesn't run on crash/force-kill
    
    2. subprocess.Popen with CREATE_SUSPENDED + enumerate threads later
       - Failed: Thread enumeration using CreateToolhelp32Snapshot took 40+ seconds
       - It is slow because it is searching all system threads for our process threads
    
    3. subprocess.Popen without suspension + immediate job assignment
       - Failed: Race condition - process may creates children before job assignment
       - Children escape the job object and survive parent death

    4. my code bellow may become shorter with pywin32, but i prefer no external dependencies, and it uses compiled dll per python version so not good for cudatext because i will have to bundle a version for each python version...
    there is also mozprocess but it recreate popen and does not expose stdout stdin and its not maintained 
    there is also processfamily but it uses pywin32 too
    
    5. in the previous PR a recreated popen using CreateProcessW, it worked fine but i do not like to touch/replace popen as it is battle tested. and i did not replicate every popen method, i reacreated just what i need for this plugin, so i will have to extend it any time i need more things...
    
    SOLUTION:
    =========
    keep using python popen, but when we use popen we have to create the process in a suspended state using CREATE_SUSPENDED because there is a tiny race condition where children created before job assignment might escape. this means we need the process or thread handle but popen do not expose them officialy. popen destroy the thread handle before returning! so one way is to enumerate threads using CreateToolhelp32Snapshot until we find our handle then we assign it to the new job object then we resume it with ResumeThread, but as i said above this took 40+ seconds

    but thanks to God I found a simpler method, python save the PROCESS handler in an undocumented private variable _handle (it is the process handle not the thread handle so we cannot use ResumeThread https://github.com/python/cpython/issues/72550, this is why in one of the previous PR's i had to use CreateToolhelp32Snapshot).
    with that handle we can then use another undocumented windows api NtResumeProcess (https://groups.google.com/g/microsoft.public.win32.programmer.kernel/c/IA-y-isvL9I) which work with the process handle so we do not need the thread handle anymore so no slowness.
    but both _handle and NtResumeProcess are undocumented so they me be removed in the future, but they existed for long time and they are used everywhere by everybody. if someday _handle is removed we can use ctypes.windll.kernel32.OpenProcess and pass the pid to it to get the handle, but it is a diferent handle we cannot use it to compare with the original one but it serves for everything else... for now i need the fastest way so i will use both of them _handle and NtResumeProcess
    
    Microsoft says:
    Windows 7, Windows Server 2008 R2, Windows XP with SP3, Windows Server 2008, Windows Vista and Windows Server 2003: A process can be associated with only one job. Jobs cannot be nested. The ability to nest jobs was added in Windows 8 and Windows Server 2012.
    Starting with Windows 8 and Windows Server 2012, an application can use nested jobs to manage a process tree that uses more than one job object. However, an application that must run on Windows 7, Windows Server 2008 R2, or earlier versions of Windows that do not support nested jobs must manage the process tree in other ways.
    Use the JOB_OBJECT_LIMIT_BREAKAWAY_OK limit. If the tool uses this limit, it can monitor the entire process tree, except for those processes that any member of the tree explicitly breaks away from the tree. A member of the tree can create a child process in a new job object by calling the CreateProcess function with the CREATE_BREAKAWAY_FROM_JOB flag, then calling the AssignProcessToJobObject function. Otherwise, the member must handle cases in which AssignProcessToJobObject fails.
    https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects
    
    so we need to use CREATE_BREAKAWAY_FROM_JOB to allow adding the process to a new job, otherwise we cannot escape win7 explorer job object which already have a job when we run the app with doble click from explorer (when calling cuda from cmd it is not needed) 

    we need also JOB_OBJECT_LIMIT_BREAKAWAY_OK, microsoft explain it again:
    Prevent breakaways of any kind by setting neither the JOB_OBJECT_LIMIT_BREAKAWAY_OK nor the JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK limit. In this option, the tool can monitor the entire process tree. However, if a child process attempts to associate itself or another child process with a job by calling AssignProcessToJobObject, the call will fail. If the process was designed to be associated with a specific job, this failure may prevent the process from working properly.
    
    so we need it to allow agents to escape from our monitoring otherwise they may fail to run if they attempt to create a new job object in win7, but this means we can no longuer kill them if cuda crash, but i did not find any agent that does this, but if this change in the future then we will have to create an external watchdog using probably an exe or batch/bash that check cudatext every 5s and if it dies then we kill the agent...
    
    DETAILS: 
    ========
    When CudaText crashes:
    1. Windows forcefully closes all handles owned by cudatext.exe
    2. This includes our job handle (h_job)
    3. Windows sees the job has no more handles -> kills all processes in job
    4. Agent and all its children die automatically
    
    This is OS-level cleanup - no Python code needs to run!
    
    RACE CONDITION PREVENTION:
    =========================
    We use CREATE_SUSPENDED to prevent race conditions:
    1. Process is created frozen (suspended)
    2. We assign it to job object while frozen
    3. Resume with NtResumeProcess (resumes all threads)
    4. No window where children can escape the job
    
    IMPLEMENTATION DETAILS:
    ======================
    - Uses subprocess.Popen for robust process creation (pipes, encoding, etc.)
    - Uses proc._handle to get Windows handle (stable since Python 2.7)
    - Uses NtResumeProcess to resume (no need to enumerate threads)
    - Stores job handle globally to prevent garbage collection (optional)
    
    PROBLEMS:
    =========
    when i run kimi agent from my agent client while i use here JOB_OBJECT_LIMIT_BREAKAWAY_OK, i get this error:
    (uv internal error) Failed to assign child process to the job: permission denied. (os error 5)

    i found in the uv source code that it does this
    if unsafe { AssignProcessToJobObject(job, child_handle) }.is_err() {
        print_last_error_and_exit("Failed to assign child process to the job")
    }
    https://github.com/astral-sh/uv/blob/c10c84a588fa91ab52e54ea353f2c928356a9e1b/crates/uv-trampoline/src/bounce.rs#L449C1-L451C6

    so uv attempts to assign kimi to a job object, by my this function also add kimi to job object, so there is a Job Object conflict on Windows.
    this occurs because of how Windows Job Objects interact when nested (or rather, when they attempt to nest) in old windows prior to win2012.
    
	1. this function puts the uv process into Job A.
	2. uv runs and spawns the kimi agent (the child).
	3. By default, children inherit the parent's Job Object. So, the agent is implicitly added to Job A.
	4. uv creates a new Job B to manage the lifecycle of the agent.
	5. uv attempts to call AssignProcessToJobObject(Job B, Agent).
	6. FAILURE: Windows blocks this because the Agent is already inside Job A. Unless Job A explicitly allows "Silent Breakaway," the Agent cannot be moved into or assigned a second job easily without specific nesting configurations that uv might not be setting up.
	
    The Solution:
    we need to change the Job Object limit flag from JOB_OBJECT_LIMIT_BREAKAWAY_OK to JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK.
    	- Current (BREAKAWAY_OK): Allows the child (uv) to break away only if it explicitly asks to (using CREATE_BREAKAWAY_FROM_JOB). uv doesn't do this when spawning its own children (i searched the source code, us do not use CREATE_BREAKAWAY_FROM_JOB).
    	- New (SILENT_BREAKAWAY_OK): Tells Windows: "Any child process spawned by the process in this job is automatically excluded from this job."
    This creates a clean chain of command:
    	1. CudaText holds Job A.
    	2. uv is inside Job A.
    	3. uv spawns Agent. Because of SILENT_BREAKAWAY, Agent is not in Job A.
    	4. uv successfully puts Agent into Job B.
    Does cleanup still work? Yes.
    If CudaText crashes > Job A kills uv > uv dies > Job B handle closes > Job B kills Agent.
        
    problem: JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK allow uv (kimi) to run, but if we use other tool and that tool spawn childrens then they will not be cleaned when cudatext crash, so there is no good solution here, windows added nested job object creation in win2012 so in win7 there no 100% good solution
    what we can do is to add a new variable called allow_zoombie which will allow the user to configure it as needed, or implement fixes for special tools like kimi so if we detect kimi we use JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK otherwise we use JOB_OBJECT_LIMIT_BREAKAWAY_OK
    
    microsoft doc:
    Use the JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK limit. If the tool uses this limit, it cannot monitor an entire process tree. The tool can monitor only the processes it adds to the job. If these processes create child processes, they are not associated with the job. In this option, child processes can be associated with other job objects.

    Use the JOB_OBJECT_LIMIT_BREAKAWAY_OK limit. If the tool uses this limit, it can monitor the entire process tree, except for those processes that any member of the tree explicitly breaks away from the tree. A member of the tree can create a child process in a new job object by calling the CreateProcess function with the CREATE_BREAKAWAY_FROM_JOB flag, then calling the AssignProcessToJobObject function. Otherwise, the member must handle cases in which AssignProcessToJobObject fails.


    ______________________________________
    some good info to remember

    https://stackoverflow.com/questions/23434842/python-how-to-kill-child-processes-when-parent-dies
    https://stackoverflow.com/questions/1884941/killing-the-children-with-the-parent

    Python: how to kill child process(es) when parent dies?

    On Linux, `prctl(PR_SET_PDEATHSIG, ...)` is probably the only reliable choice. (If it's absolutely necessary that the child process be killed, then you might want to set the death signal to SIGKILL instead of SIGTERM; the code you linked to uses SIGTERM, but the child does have the option of ignoring SIGTERM if it wants to.)

    On Windows, the most reliable options is to use a [Job object](http://msdn.microsoft.com/en-us/library/ms684161%28VS.85%29.aspx). The idea is that you create a "Job" (a kind of container for processes), then you place the child process into the Job, and you set the magic option that says "when no-one holds a 'handle' for this Job, then kill the processes that are in it". By default, the only 'handle' to the job is the one that your parent process holds, and when the parent process dies, the OS will go through and close all its handles, and then notice that this means there are no open handles for the Job. So then it kills the child, as requested. (If you have multiple child processes, you can assign them all to the same job.) [This answer](https://stackoverflow.com/a/12942797/1925449) has sample code for doing this, using the `win32api` module. That code uses `CreateProcess` to launch the child, instead of `subprocess.Popen`. The reason is that they need to get a "process handle" for the spawned child, and `CreateProcess` returns this by default. If you'd rather use `subprocess.Popen`, then here's an (untested) copy of the code from that answer, that uses `subprocess.Popen` and `OpenProcess` instead of `CreateProcess`:

    # Source - https://stackoverflow.com/a/23587108
    # Posted by Nathaniel J. Smith, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-01-07, License - CC BY-SA 3.0

    import subprocess
    import win32api
    import win32con
    import win32job

    hJob = win32job.CreateJobObject(None, "")
    extended_info = win32job.QueryInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation)
    extended_info['BasicLimitInformation']['LimitFlags'] = win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    win32job.SetInformationJobObject(hJob, win32job.JobObjectExtendedLimitInformation, extended_info)

    child = subprocess.Popen(...)
    # Convert process id to process handle:
    perms = win32con.PROCESS_TERMINATE | win32con.PROCESS_SET_QUOTA
    hProcess = win32api.OpenProcess(perms, False, child.pid)

    win32job.AssignProcessToJobObject(hJob, hProcess)


    Technically, there's a tiny race condition here in case the child dies in between the `Popen` and `OpenProcess` calls, you can decide whether you want to worry about that.

    One downside to using a job object is that when running on Vista or Win7, if your program is launched from the Windows shell (i.e., by clicking on an icon), then there will probably [already be a job object assigned](https://social.msdn.microsoft.com/Forums/windowsdesktop/en-US/71c9599e-a3d5-4b79-bfc1-1800565c5b8a/assignprocesstojobobject-always-return-access-denied-on-vista?forum=windowssecurity) and trying to create a new job object will fail. Win8 fixes this (by allowing job objects to be nested), or if your program is run from the command line then it should be fine.

    If you _can_ modify the child (e.g., like when using `multiprocessing`), then probably the best option is to somehow pass the parent's PID to the child (e.g. as a command line argument, or in the `args=` argument to `multiprocessing.Process`), and then:

    On POSIX: Spawn a thread in the child that just calls `os.getppid()` occasionally, and if the return value ever stops matching the pid passed in from the parent, then call `os._exit()`. (This approach is portable to all Unixes, including OS X, while the `prctl` trick is Linux-specific.)

    On Windows: Spawn a thread in the child that uses `OpenProcess` and `os.waitpid`. Example using ctypes:

    # Source - https://stackoverflow.com/a/23587108
    # Posted by Nathaniel J. Smith, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-01-07, License - CC BY-SA 3.0

    from ctypes import WinDLL, WinError
    from ctypes.wintypes import DWORD, BOOL, HANDLE
    # Magic value from http://msdn.microsoft.com/en-us/library/ms684880.aspx
    SYNCHRONIZE = 0x00100000
    kernel32 = WinDLL("kernel32.dll")
    kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
    kernel32.OpenProcess.restype = HANDLE
    parent_handle = kernel32.OpenProcess(SYNCHRONIZE, False, parent_pid)
    # Block until parent exits
    os.waitpid(parent_handle, 0)
    os._exit(0)


    This avoids any of the possible issues with job objects that I mentioned.

    If you want to be really, really sure, then you can combine all these solutions.
    _______________________
    some usefull comments
    _______________________

    One way of avoiding the race condition you mention is to add yourself to the job object before launching the child process; the child will inherit membership. Another way is to launch the child process suspended, and only resume it after adding it to the job.
    _
    For Windows 7, the shell's job object allows breaking away, so you can use the creation flag CREATE_BREAKAWAY_FROM_JOB to allow adding the process to a new job.
    _
    It might be obvious to many, but to actually terminate, one needs to add win32job.TerminateJobObject(hJob, hProcess) after the process is no longer needed.
    _
    Besides the race condition problem, it looks like there is also the possibility of a deadlock if any threads are used in the Python program since prcrtl is not a signal-safe system function.

    _________________________
    
    https://stackoverflow.com/questions/34551464/resume-thread-of-process-created-suspended-having-lost-the-thread-handle
    Resume thread of process created suspended, having lost the thread handle
    On Windows, Python (2)'s standard library routine subprocess.Popen allows you to specify arbitrary flags to CreateProcess, and you can access the process handle for the newly-created process from the object that Popen returns. However, the thread handle for the newly-created process's initial thread is closed by the library before Popen returns.
    Now, I need to create a process suspended (CREATE_SUSPENDED in creation flags) so that I can manipulate it (specifically, attach it to a job object) before it has a chance to execute any code. However, that means I need the thread handle in order to release the process from suspension (using ResumeThread). The only way I can find, to recover the thread handle, is to use the "tool help" library to walk over all threads on the entire system (e.g. see this question and answer https://stackoverflow.com/questions/9965784/how-to-obtain-list-of-thread-handles-from-a-win32-process). This works, but I do not like it. Specifically, I am concerned that taking a snapshot of all the threads on the system every time I need to create a process will be too expensive. (The larger application is a test suite, using processes for isolation; it creates and destroys processes at a rate of tens to hundreds a second.)
    So, the question is: is there a more efficient way to resume execution of a process that was suspended by CREATE_SUSPENDED, if all you have is the process handle, and the facilities of the Python 2 standard library (including ctypes, but not the winapi add-on)? Vista-and-higher techniques are acceptable, but XP compatibility is preferred.

    _
    I have found a faster approach; unfortunately it relies on an undocumented API, NtResumeProcess. This does exactly what it sounds like - takes a process handle and applies the equivalent of ResumeThread to every thread in the process. Python/ctypes code to use it looks something like
    ...
    I measured approximately 20% less process setup overhead using this technique than using Toolhelp, on an otherwise-idle Windows 7 virtual machine. As expected given how Toolhelp works, the performance delta gets bigger the more threads exist on the system -- whether or not they have anything to do with the program in question.
    Given the obvious general utility of NtResumeProcess and its counterpart NtSuspendProcess, I am left wondering why they have never been documented and given kernel32 wrappers. They are used by a handful of core system DLLs and EXEs all of which, AFAICT, are part of the Windows Error Reporting mechanism (faultrep.dll, werui.dll, werfault.exe, dwwin.exe, etc) and don't appear to re-expose the functionality under documented names. It seems unlikely that these functions would change their semantics without also changing their names, but a defensively-coded program should probably be prepared for them to disappear (falling back to toolhelp, I suppose).

    _________________________

    https://stackoverflow.com/questions/9965784/how-to-obtain-list-of-thread-handles-from-a-win32-process
    How to obtain list of thread handles from a win32 process?
    Just be aware that OpenThread returns an alternate HANDLE, not the original HANDLE. It is useful for everything except comparing to the original handle. And don't forget to CloseHandle when you are done with this alternate HANDLE.
    _________________________
    
    https://stackoverflow.com/questions/76876658/stopping-children-being-terminated-when-parent-process-was-started-in-a-job-i-ca
    Stopping children being terminated when parent process was started in a Job I can't control
    I have a process (A) that I don't control that starts my own developed process (B) which in turn will start several other processes (C).

    It looks like A starts B using a Job object as when A is terminated B is also terminated. I've also confirmed this with Process Explorer. The issue I am having is that the child processes of B are also being terminated, which is normal behavior for Job objects.

    Is there a way for me to launch child processes from B that don't inherit the Job attributes or have a new set of attributes?
    _
    If Process B is starting Process C via CreateProcess(), then Process B can specify the CREATE_BREAKAWAY_FROM_JOB flag:
    But, as stated, this only works if the Job has the JOB_OBJECT_LIMIT_BREAKAWAY_OK limit enabled to allow child processes to "break away" from the job:

    If Process A does not enable the `JOB_OBJECT_LIMIT_BREAKAWAY_OK` limit, then it doesn't want break aways. Process B _might_ be able to update that, but only if it knows the name of the Job that it is running in so it can obtain a `HANDLE` to that Job:

    [IsProcessInJob function](https://learn.microsoft.com/en-us/windows/win32/api/jobapi/nf-jobapi-isprocessinjob)

    > An application cannot obtain a handle to the job object in which it is running unless it has the name of the job object. However, an application can call the [QueryInformationJobObject](https://learn.microsoft.com/en-us/windows/desktop/api/jobapi2/nf-jobapi2-queryinformationjobobject) function with NULL to obtain information about the job object.
    
    ___
    https://stackoverflow.com/questions/52112113/can-a-child-process-which-was-started-by-a-process-in-job-set-job-properties-to
    Assuming the parent process is still running and has a handle for the job, the child could enumerate all handles in the system (or just the parent process via PssCaptureSnapshot in Windows 8.1+); duplicate each job handle to itself; check whether it's in the job via IsProcessInJob; and if so, and if allowed, modify the job to allow breakaway. If successful, the program can subsequently break free by respawning itself as a new process.
    
    """

    # Prepare environment - merge with os.environ, giving priority to provided env
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    
    if sys.platform == 'win32':
        import ctypes
        from ctypes import wintypes
        
        kernel32 = ctypes.windll.kernel32
        ntdll = ctypes.windll.ntdll
        
        # Windows structures for job object configuration
        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('PerProcessUserTimeLimit', ctypes.c_int64),
                ('PerJobUserTimeLimit', ctypes.c_int64),
                ('LimitFlags', ctypes.c_uint32),
                ('MinimumWorkingSetSize', ctypes.c_size_t),
                ('MaximumWorkingSetSize', ctypes.c_size_t),
                ('ActiveProcessLimit', ctypes.c_uint32),
                ('Affinity', ctypes.POINTER(ctypes.c_ulong)),
                ('PriorityClass', ctypes.c_uint32),
                ('SchedulingClass', ctypes.c_uint32),
            ]
        
        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [
                ('ReadOperationCount', ctypes.c_uint64),
                ('WriteOperationCount', ctypes.c_uint64),
                ('OtherOperationCount', ctypes.c_uint64),
                ('ReadTransferCount', ctypes.c_uint64),
                ('WriteTransferCount', ctypes.c_uint64),
                ('OtherTransferCount', ctypes.c_uint64),
            ]
        
        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [
                ('BasicLimitInformation', JOBOBJECT_BASIC_LIMIT_INFORMATION),
                ('IoInfo', IO_COUNTERS),
                ('ProcessMemoryLimit', ctypes.c_size_t),
                ('JobMemoryLimit', ctypes.c_size_t),
                ('PeakProcessMemoryUsed', ctypes.c_size_t),
                ('PeakJobMemoryUsed', ctypes.c_size_t),
            ]
        
        # Windows constants
        CREATE_SUSPENDED = 0x00000004
        CREATE_BREAKAWAY_FROM_JOB = 0x01000000
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
        JOB_OBJECT_LIMIT_BREAKAWAY_OK = 0x800
        JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x1000
        
        # Step 1: Create job object
        h_job = kernel32.CreateJobObjectW(None, None)
        if not h_job:
            raise Exception(f"CreateJobObject failed: {kernel32.GetLastError()}")
        
        debug_print("Job object created")
        
        # Step 2: Configure job to kill processes when handle closes and Configure job limit flags based on allow_childs_of_child
        limit_flags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        
        if allow_childs_of_child:
            # Silent breakaway allows the childs of the child process to automatically break away from the job.
            # This allows the child (e.g. uv) to create its own job object for its children in older windows 7.
            # this allow childrens of children to break away always, with this we monitor only the processes we adds to the job. If these processes create child processes, they are not associated with the job.
            limit_flags |= JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK
        else:
            # Standard behavior: Allow childrens of children to break away if they ask to. this allow breakaway only when the child explicitly use CREATE_BREAKAWAY_FROM_JOB flag. otherwise keep everything controled by us.
            limit_flags |= JOB_OBJECT_LIMIT_BREAKAWAY_OK

        job_info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        job_info.BasicLimitInformation.LimitFlags = limit_flags
        
        if not kernel32.SetInformationJobObject(
            h_job,
            9,  # JobObjectExtendedLimitInformation
            ctypes.byref(job_info),
            ctypes.sizeof(job_info)
        ):
            kernel32.CloseHandle(h_job)
            raise Exception(f"SetInformationJobObject failed: {kernel32.GetLastError()}")
        
        debug_print("Job configured with KILL_ON_JOB_CLOSE")
        
        # Step 3: Create process in SUSPENDED state using subprocess.Popen
        # This gives us robust pipe handling while preventing race conditions
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        
        # Create process suspended with breakaway and hidden console window
        try:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',  # Explicitly tell Python the stream is UTF-8, claude send its models desc with some utf8 char
                errors='replace',   # Optional: prevents crashes if there's a stray bad byte
                bufsize=1,
                cwd=cwd,
                env=process_env,
                startupinfo=startupinfo,
                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP | 
                              subprocess.CREATE_NO_WINDOW |
                              CREATE_SUSPENDED |
                              CREATE_BREAKAWAY_FROM_JOB)
            )
            
            debug_print(f"Process created suspended with PID: {process.pid}")
            
        except Exception as e:
            kernel32.CloseHandle(h_job)
            raise
        
        # Step 4: Assign process to job using _handle BEFORE resuming
        # proc._handle is the Windows HANDLE that subprocess.Popen stores
        # It's been stable since Python 2.7 and is the standard undocumented way to access it
        if not kernel32.AssignProcessToJobObject(h_job, int(process._handle)):
            error = kernel32.GetLastError()
            kernel32.CloseHandle(h_job)
            process.kill()
            raise Exception(f"AssignProcessToJobObject failed: {error}")
        
        debug_print("Process assigned to job object")
        
        # Step 5: Resume process using NtResumeProcess
        # NtResumeProcess resumes ALL threads in the process at once
        # No need to enumerate threads like with ResumeThread
        status = ntdll.NtResumeProcess(int(process._handle))
        if status == 0:
            debug_print("Process resumed successfully")
        else:
            debug_print(f"NtResumeProcess returned status: {status}")
        
        # Step 6: Store job handle globally to prevent garbage collection
        # If h_job is GC'd, Windows closes the handle and kills all processes in job
        # this is usefull if we want processes to die ONLY when CudaText crashes, and we will not keep a reference of the handle  (for example we want to run a tool and forget it but kill it once cudatext die), create a global variable _ACTIVE_JOB_HANDLES and uncomment this. this is not needed for this plugin because we already keep a reference to the handle in process.job_handle and we want the ai agent to die once we no longuer keep a ref to the handler for example when doing "process.job_handle = None"
        # _ACTIVE_JOB_HANDLES.append(h_job)
        # debug_print(f"Job handle stored globally (total active jobs: {len(_ACTIVE_JOB_HANDLES)})")
        
        # Store job handle on process for manual cleanup if needed and to prevent garbage collection
        process.job_handle = h_job
        
        debug_print(f"Process {process.pid} created with job object")
        
        return process
        
    else:
        # Unix: use PR_SET_PDEATHSIG to kill process when parent dies
        
        # Use prctl with PR_SET_PDEATHSIG to send SIGKILL to child when parent dies.
        # This is a Linux-specific feature that works at kernel level.
        # 
        # For portability, we use preexec_fn which:
        # 1. Creates new session (setsid) - detaches from terminal
        # 2. Sets up signal handler via prctl (Linux only)
        # 
        # This ensures children die when CudaText crashes or is force-killed.
        
        import signal
        
        def preexec():
            """Setup child to die when parent dies (Linux-specific)"""
            # Create new session
            os.setsid()
            # Linux-specific: send SIGKILL to this process when parent dies
            try:
                import ctypes
                import ctypes.util
                libc = ctypes.CDLL(ctypes.util.find_library('c'))
                PR_SET_PDEATHSIG = 1
                # Send SIGKILL to this process when parent dies
                libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL)
            except:
                # prctl not available on non-Linux Unix, best effort
                pass
        
        return subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',  # Explicitly tell Python the stream is UTF-8, claude send its models desc with some utf8 char
            errors='replace',   # Optional: prevents crashes if there's a stray bad
            bufsize=1,
            cwd=cwd,
            env=process_env,
            preexec_fn=preexec
        )
        
    
# =============================================================================
# DEBUG CONFIGURATION
# =============================================================================
DEBUG = False  # Set to True to enable debug output, too much verbose, DEBUG_SPY is sufficient except in extreme cases

DEBUG_SPY = True  # Set to True to enable RPC communication spy panel, lets keep it always True now that we added the gui menu the show/hide it and config setting to restore the state, so do not set it to false otherwise the gui menu will not work

# Global variable to track the last time debug_print was called
if DEBUG:
    _last_debug_time = time.perf_counter()

def debug_print(*args, **kwargs):
    """Print debug messages with timestamp and delta when DEBUG is enabled
    RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
    """
    global _last_debug_time
    if DEBUG:
        now = time.strftime("%H:%M:%S")
        current_perf = time.perf_counter()
        delta = current_perf - _last_debug_time
        
        # Example Output: [AI DEBUG] [14:30:05] [+0.0123s] message
        print(f"[AI DEBUG] [{now}] [+{delta:.4f}s]", *args, **kwargs)
        _last_debug_time = current_perf

# =============================================================================
# Agent configurations for various ACP-compatible agents
# =============================================================================

# List of supported agents
AGENTS: List[Dict[str, Any]] = [

    {
        'id': 'gemini',
        'name': 'Gemini',
        'command': 'gemini',
        'args': ['--experimental-acp'],
        'supported_models': [
            {'id': 'gemini-2.5-pro', 'name': 'Gemini 2.5 Pro'},
            {'id': 'gemini-2.5-flash', 'name': 'Gemini 2.5 Flash'},
            {'id': 'gemini-2.5-flash-lite', 'name': 'Gemini 2.5 Flash Lite'},
            {'id': 'gemini-3-pro', 'name': 'Gemini 3 Pro'},
            {'id': 'gemini-3-flash', 'name': 'Gemini 3 Flash'},
        ]
    },
    
    {
        'id': 'opencode',
        'name': 'OpenCode',
        'command': 'opencode',
        'args': ['acp']
    },
    {
        'id': 'claude-code',
        'name': 'Claude',
        'link': 'https://github.com/zed-industries/claude-code-acp',
        'command': 'claude-code-acp',
    },
    {
        'id': 'codex',
        'name': 'Codex',
        'link': 'https://github.com/zed-industries/codex-acp',
        'command': 'codex-acp'
    },
    
    {
        'id': 'goose',
        'name': 'Goose',
        'command': 'goose',
        'args': ['acp']
    },
    # {
        # 'id': 'amp',
        # 'name': 'Amp',
        # 'command': 'amp',
        # 'args': ['acp']
    # },
    {
        # https://github.com/tao12345666333/amp-acp
        'id': 'amp-acp',
        'name': 'Ampcode',
        'command': 'amp-acp'
    },
    {
        'id': 'aider',
        'name': 'Aider',
        'command': 'aider',
        'args': ['--acp']
    },
    {
        'id': 'augment',
        'name': 'Augment',
        'command': 'augment',
        'args': ['acp']
    },
    {
        'id': 'kimi',
        'name': 'Kimi',
        'command': 'kimi',
        'args': ['acp']
    },
    {
        # https://github.com/mistralai/mistral-vibe
        'id': 'mistral-vibe',
        'name': 'Mistral Vibe',
        'command': 'vibe-acp',
    },
    {
        'id': 'openhands',
        'name': 'OpenHands',
        'command': 'openhands',
        'args': ['acp']
    },
    {
        'id': 'qwen',
        'name': 'Qwen Code',
        'command': 'qwen',
        'args': ['--acp']
    },
    {
        'id': 'iflow',
        'name': 'Iflow',
        'command': 'iflow',
        'args': ['--experimental-acp']
    },
    {
        'id': 'llxprt',
        'name': 'Llxprt',
        'command': 'llxprt',
        'args': ['--experimental-acp']
    }
]

# there is something wrong with threads inside cudatext, when i run shutil.which inside connect() which runs inside a thread it takes 30s!!!, this does not happen with threads in general! i could not find why this happen, so i cache the shutil.which result here outside the threads to solve it
# Cache executables AFTER defining AGENTS but BEFORE using get_agents_info
EXEC_CACHE = {}

# First, cache from default AGENTS list
for agent in AGENTS:
    cmd = agent.get('command')
    if cmd:
        exe = shutil.which(cmd)
        EXEC_CACHE[cmd] = exe

# Then load config and cache any additional commands from user-defined agents
__, config = load_and_merge_config()
for agent_id, agent_config in config.get('agent_servers', {}).items():
    cmd = agent_config.get('command')
    if cmd and cmd not in EXEC_CACHE:
        exe = shutil.which(cmd)
        EXEC_CACHE[cmd] = exe

debug_print("Cached executables:", EXEC_CACHE)

def get_agents_info(return_first_available=False):
    """Get agents with status or return first available agent"""
    agents_with_status = []
    first_available = None
    
    for agent in AGENTS:
        agent_copy = agent.copy()
        agent_copy['available'] = EXEC_CACHE.get(agent['command']) is not None
        agents_with_status.append(agent_copy)
        
        if return_first_available and first_available is None and agent_copy['available']:
            first_available = agent_copy
    
    if return_first_available:
        return first_available or AGENTS[0]
    return agents_with_status

# =============================================================================
# ACP Client - Handles connection to ACP agents via subprocess
# =============================================================================

class ACPClient:
    """Client for communicating with ACP agents via JSON-RPC 2.0"""
    
    def __init__(self, agent_config: Dict[str, Any], message_queue: Queue):
        self.agent_config = agent_config
        self.message_queue = message_queue
        self.process: Optional[subprocess.Popen] = None
        self.job_handle = None  # Windows job object handle
        self.state = 'disconnected'
        self.session_id: Optional[str] = None
        self.session_metadata: Optional[Dict[str, Any]] = None
        
        # Session loading capability
        self.supports_load_session = False

        # authentication
        self.auth_methods: List[Dict[str, Any]] = []
        self.authenticated = False
                
        # Message queues for thread communication
        self.outgoing_queue = Queue()  # Messages to send to agent
        self.incoming_queue = Queue()  # Messages received from agent
        self.request_id = 0
        self.pending_requests: Dict[int, Queue] = {}  # Maps request ID to response queue
        
        # Threading
        self.read_thread: Optional[threading.Thread] = None
        self.write_thread: Optional[threading.Thread] = None
        self.stderr_thread: Optional[threading.Thread] = None
        self.should_stop = False
        
        debug_print(f"ACPClient initialized with agent: {agent_config['name']}")
        
    def _spy_log(self, direction: str, msg_type: str, data: Any, request_id: Any = None):
        """
        Log RPC communication to spy panel
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        if not DEBUG_SPY:
            return
            
        now = time.strftime("%H:%M:%S")
        
        # Format message type with icons
        if direction == 'send':
            if msg_type == 'request':
                prefix = f"[{now}] --► SEND REQUEST    "
            elif msg_type == 'notification':
                prefix = f"[{now}]  -► SEND NOTIFICATION"
            elif msg_type == 'response':
                prefix = f"[{now}] ►►► SEND RESPONSE   "
        elif direction == 'recv':
            if msg_type == 'response':
                prefix = f"[{now}] ◄◄◄ RECV RESPONSE   "
            elif msg_type == 'notification':
                prefix = f"[{now}] ◄-  RECV NOTIFICATION"
            elif msg_type == 'request':
                prefix = f"[{now}] ◄-- RECV REQUEST    "
            elif msg_type == 'error':
                prefix = f"[{now}] ███ RECV ERROR      "
        elif direction == 'stderr':
            prefix = f"[{now}] ΔΔΔ STDERR          "
        else:
            prefix = f"[{now}] ??? {direction} "
        
        # Build message info
        info = ""
        payload = ""
        
        if isinstance(data, dict):
            method = data.get('method', '')
            if request_id is not None:
                info = f": {request_id}"
            elif method:
                info = f": {method}"
            try:
                payload = json.dumps(data)
            except:
                payload = str(data)
        else:
            # Handle string data (like stderr lines)
            payload = str(data).strip()
        
        # Format the full message
        msg_str = f"{prefix}{info} _ {payload}\n"
        
        # Queue spy message for main thread processing
        self.message_queue.put(('spy', msg_str))
                
    def establish_agent_connection(self):
        """
        Connect to the ACP agent following the initialization protocol
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print("=== CONNECTING TO AGENT ===")
        
        if self.state == 'connected':
            raise Exception('Already connected')
        elif self.state == 'connecting':
            raise Exception('Already connecting')
            
        self.message_queue.put(('connection_state_changed', 'connecting'))
        
        try:
            # 1. Resolve executable path
            cmd_name = self.agent_config.get('command')
            executable = EXEC_CACHE.get(cmd_name)
            if not executable:
                raise Exception(f"Command not found: {cmd_name}")

            debug_print(f"Executable found: {executable}")
            
            # 2. Build command list safely
            # Only add args if they exist in config and are a list
            cmd = [executable]
            extra_args = self.agent_config.get('args')
            if isinstance(extra_args, list):
                cmd.extend(extra_args)
            
            debug_print(f"Starting process: {' '.join(cmd)}")
            
            # 3. Start process with cleanup logic and without CMD window
            ''' met1: may leave zombie processes
            if sys.platform == 'win32':
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE
                
                self.process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                # On Unix-like systems, no special flags needed
                self.process = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
            '''
            
            # met2: Create process with automatic cleanup (job objects on Windows, prctl on Linux)
            # Get config settings to check for allowed child processes for kimi...etc (to fix uv permission error)
            config_settings = get_config_settings()
            allowed_childs_agents = config_settings.get('servers_id_with_allowed_childs', [])

            agent_id = self.agent_config.get('id', '').lower()
            allow_childs = agent_id in allowed_childs_agents

            # Prepare environment variables from agent config
            agent_env = self.agent_config.get('env', {})

            # Create process with env and child process settings
            self.process = create_process_with_cleanup(
                cmd, 
                cwd=None,
                allow_childs_of_child=allow_childs,
                env=agent_env if agent_env else None
            )            

            # Store job handle separately for cleanup
            self.job_handle = self.process.job_handle
            
            debug_print("Process started successfully")
            
            # 4. Start reader/writer threads
            self.should_stop = False
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.write_thread = threading.Thread(target=self._write_loop, daemon=True)
            self.stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
            
            self.read_thread.start()
            self.write_thread.start()
            self.stderr_thread.start()
            
            
            # FORCE SCHEDULER YIELD
            # This gives the threads a chance to actually boot up before the 
            # main thread starts hammering the I/O with JSON requests
            # Optional: Short sleep to reduce contention (empirical - helps scheduler on some Windows setups)
            # without this sleep starting above threads take 30 to 50s!! another bug in threads inside cudatext? who knows!
            time.sleep(0.1)  # Let threads initialize
            
            debug_print("Communication threads started")
            
            # Step 1: Send initialize request
            debug_print("Step 1: Sending initialize request")
            init_response = self._send_request_and_wait_for_response('initialize', {
                'protocolVersion': 1,
                'clientCapabilities': {
                    'terminal': False,
                    'fs': {
                        'readTextFile': False,
                        'writeTextFile': False
                    }
                },
                'clientInfo': {
                    'name': 'cudatext-ai-agents',
                    'version': '1.0.0'
                }
            })
            
            debug_print(f"Initialize response: {json.dumps(init_response)}")
            
            # Check if agent supports session loading
            agent_capabilities = init_response.get('agentCapabilities', {})
            self.supports_load_session = agent_capabilities.get('loadSession', False)
            debug_print(f"Agent supports loadSession: {self.supports_load_session}")
            
            # Store modes from initialize response (if provided)
            if 'modes' in init_response:
                if not self.session_metadata:
                    self.session_metadata = {}
                self.session_metadata['modes'] = init_response['modes']
                debug_print(f"Modes from initialize: {json.dumps(init_response['modes'])}")
            
            # Store auth methods if provided
            self.auth_methods = init_response.get('authMethods', [])
                        
            self.message_queue.put(('connection_state_changed', 'connected'))

            debug_print("=== CONNECTION ESTABLISHED ===")
            return init_response
            
        except Exception as e:
            debug_print(f"Connection error: {e}")
            self.message_queue.put(('connection_state_changed', 'error'))
            if DEBUG:
                import traceback
                traceback.print_exc()
            if self.process:
                # kill
                try:
                    self.process.terminate()
                except:
                    pass
                # Cleanup Windows Job Object handle manually on error on Windows
                if sys.platform == 'win32' and hasattr(self.process, 'job_handle') and self.process.job_handle:
                    try:
                        import ctypes
                        ctypes.windll.kernel32.CloseHandle(self.process.job_handle)
                    except:
                        pass
                self.process = None
                self.job_handle = None
            raise
      
    def new_session(self, working_directory: str) -> Dict[str, Any]:
        """
        Create a new session
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print(f"=== CREATING NEW SESSION (cwd: {working_directory}) ===")
        
        if not self.process:
            raise Exception('Not connected')
        
        # Send session/new request
        response = self._send_request_and_wait_for_response('session/new', {
            'cwd': working_directory,
            'mcpServers': []
        })
        
        self.session_id = response.get('sessionId')
        
        # Initialize session_metadata if not exists
        if not self.session_metadata:
            self.session_metadata = {}
        
        # Merge modes from session/new response (takes precedence over initialize)
        if 'modes' in response:
            self.session_metadata['modes'] = response['modes']
            debug_print(f"Modes from session/new: {json.dumps(response['modes'])}")
        
        # Store models and commands from session/new
        if 'models' in response:
            self.session_metadata['models'] = response['models']
            debug_print(f"Models from session/new: {json.dumps(response['models'])}")
                    
        debug_print(f"Session created with ID: {self.session_id}")
        debug_print(f"Final session metadata: {json.dumps(self.session_metadata)}")

        return response

    def _authenticate(self):
        """
        Handle authentication flow - let user choose auth method
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print("\n=== Authentication Required ===")
        
        if not self.auth_methods:
            raise Exception("No authentication methods available")
        
        # Build menu items from auth methods
        menu_items = []
        for idx, method in enumerate(self.auth_methods):
            name = method.get('name', method['id'])
            description = method.get('description', '')
            
            # Format display text
            if description:
                display = f"{idx + 1}. {name}\t    {description}"
            else:
                display = f"{idx + 1}. {name}"
            
            menu_items.append(display)
            debug_print(f"{idx + 1}. {method['id']}: {name}")
        
        # Queue request to show menu in main thread
        response_queue = Queue()
        self.message_queue.put(('auth_method_selection', {
            'methods': self.auth_methods,
            'menu_items': menu_items,
            'response_queue': response_queue
        }))
        
        # Wait for user selection
        try:
            selected_method_id = response_queue.get(timeout=300)  # 5 min timeout
            
            if not selected_method_id:
                raise Exception("Authentication cancelled by user")
            
            debug_print(f"Selected auth method: {selected_method_id}")
            
            # Send authentication request
            response = self._send_request_and_wait_for_response('authenticate', {'methodId': selected_method_id})
            self.authenticated = True
            debug_print("✓ Authentication successful!")
            
        except Empty:
            raise Exception("Authentication selection timed out")
        
        # ==============================
        # here we manage non compliant ACP agents that do not support ACP authentication method like claude, lets show a usefull message
        # ==============================
        except Exception as e:
            # Check if this is a "method not implemented" error (-32603)
            if hasattr(e, 'error_code') and e.error_code == -32603:
                # Get the selected method details
                selected_method = None
                for method in self.auth_methods:
                    if method['id'] == selected_method_id:
                        selected_method = method
                        break
                
                if selected_method:
                    method_name = selected_method.get('name', selected_method_id)
                    description = selected_method.get('description', '')
                    agent_name = self.agent_config.get('name', 'Agent')
                    
                    # Build helpful error message
                    error_msg = f"Authentication method '{method_name}' requires manual setup.\n\n"
                    
                    if description:
                        error_msg += f"Instructions: {description}\n\n"
                    
                    error_msg += f"Steps to authenticate {agent_name}:\n"
                    error_msg += "1. Open a terminal/command prompt\n"
                    error_msg += f"2. Run the authentication command shown in the instructions\n"
                    error_msg += "3. Complete the authentication process\n"
                    error_msg += "4. Return to CudaText and try connecting again\n\n"
                    error_msg += "The agent should then work without requiring authentication."
                    
                    # Queue helpful message to UI
                    self.message_queue.put(('auth_manual_setup_required', {
                        'message': error_msg,
                        'method_name': method_name,
                        'description': description
                    }))
                    
                    raise Exception(f"Manual authentication required: {description if description else method_name}")
                
            # Re-raise other errors
            raise
        # ==============================
        # end of non ACP compliant fix
        # ==============================


    def send_prompt_to_agent(self, message: str) -> Dict[str, Any]:
        """
        Send a message to the agent
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print(f"=== SENDING MESSAGE: {message[:50]}... ===")
        
        if not self.session_id:
            raise Exception('No active session')
            
        return self._send_request_and_wait_for_response('session/prompt', {
            'sessionId': self.session_id,
            'prompt': [{'type': 'text', 'text': message}]
        })
        
    def set_session_mode(self, mode_id: str):
        """
        Set session mode
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print(f"Setting mode to: {mode_id}")
        
        if not self.session_id:
            raise Exception('No active session')
            
        self._send_request_and_wait_for_response('session/set_mode', {
            'sessionId': self.session_id,
            'modeId': mode_id
        })
        
        # Update local metadata
        if self.session_metadata and 'modes' in self.session_metadata:
            modes = self.session_metadata['modes']
            if modes:
                modes['currentModeId'] = mode_id
                debug_print(f"Updated current mode to: {mode_id}")

    def set_session_model(self, model_id: str):
        """
        Set session model
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print(f"Setting model to: {model_id}")
        
        if not self.session_id:
            raise Exception('No active session')
            
        self._send_request_and_wait_for_response('session/set_model', {
            'sessionId': self.session_id,
            'modelId': model_id
        })
        
        if self.session_metadata and 'models' in self.session_metadata:
            models = self.session_metadata['models']
            if models:
                models['currentModelId'] = model_id
                        
    def _send_notification(self, method: str, params: Dict[str, Any]):
        """
        Send a JSON-RPC notification (fire and forget)
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        message = {
          'jsonrpc': '2.0',
          'method': method,
          'params': params
        }
        
        debug_print(f">>> SENDING NOTIFICATION: {method}")
        debug_print(json.dumps(message))
        self._spy_log('send', 'notification', message)
        self.outgoing_queue.put(message)

    def _send_request_and_wait_for_response(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a JSON-RPC request and wait for response (blocks current thread)
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        
        Auto-retries once on authentication error (-32000)
        
        WAITING STRATEGY:
        ===================================
        This method implements a robust waiting strategy that handles:
        
        1. INDEFINITE WAITING: No fixed timeout - agent can "think" for hours if needed
        
        2. PERIODIC SAFETY CHECKS: Polls every 1 second to detect:
           - Process crashes (agent died)
           - Client shutdown (user clicked disconnect/kill)
           - Ensures we don't hang forever on dead connections
        
        3. SAFETY VALVE: The _cleanup_state method also puts error messages in all pending queues when disconnect/kill is called, providing redundant protection
        
        4. CONSISTENT ERROR FORMAT: All errors (crashes, timeouts, cancellations) are delivered as JSON-RPC error objects in the queue for uniform handling
        
        WHY THIS APPROACH:
        ==================
        - Queue-based (not Event-based): Natural fit for RPC - we're transferring data
        - Polling (not infinite block): Detects crashes immediately without relying on cleanup
        - Short timeout (1s): Low overhead, responsive to crashes, not a busy loop
        - Two-layer protection: Periodic checks + safety valve = robust
        
        ALTERNATIVE APPROACHES CONSIDERED:
        ==================================
        1. Fixed 30s timeout: Fails for slow reasoning models
        2. Infinite wait (timeout=None): Hangs forever if process crashes
        3. Event-based: More complex, requires Event + container, no practical benefit
        
        Args:
            - method: JSON-RPC method name (e.g., 'session/new', 'session/prompt')
            - params: Method parameters as dictionary
        
        Returns:
            Response result dict for requests, None for notifications
        
        Raises:
            Exception: For JSON-RPC errors, process crashes, or client shutdown
        """
        auth_attempted = False
        
        while True:
            self.request_id += 1
            req_id = self.request_id
            message = {
              'jsonrpc': '2.0',
              'id': req_id,
              'method': method,
              'params': params
            }
            
            debug_print(f">>> SENDING REQUEST #{req_id}: {method}")
            debug_print(json.dumps(message))
            self._spy_log('send', 'request', message, req_id)
            
            # Create response queue for this specific request
            response_queue = Queue()
            self.pending_requests[req_id] = response_queue
            
            # Send request to agent
            self.outgoing_queue.put(message)
            
            # ============================================================
            # WAITING LOOP - Core of the waiting strategy
            # ============================================================
            try:
                response = None
                
                while True:
                    try:
                        # Wait for response with SHORT timeout (1 second). This allows us to check safety conditions periodically. without busy-waiting or hanging forever
                        response = response_queue.get(timeout=1.0)
                    
                        # Got a response! Break out of loop to process it
                        break
                        
                    except Empty:
                        # Timeout - no response yet. This is NORMAL and expected.
                        # Now perform safety checks before continuing to wait.
                        
                        # SAFETY CHECK 1: Is the agent process still alive?
                        # ================================================
                        # If process crashed/terminated, we should stop waiting
                        # and notify the caller with an error.
                        # 
                        # self.process.poll() returns:
                        #   - None if process is still running
                        #   - exit code (int) if process has terminated
                        if not self.process or self.process.poll() is not None:
                            debug_print(f"Process died while waiting for response #{req_id}")
                        
                            # Put error in queue so it's processed the same way
                            # as real responses (consistent error handling)
                            response_queue.put({
                                'error': {
                                    'code': -70000,  # Internal error
                                    'message': 'Agent process terminated unexpectedly'
                                }
                            })
                            # Continue to next iteration to retrieve this error
                            continue
                        
                        # SAFETY CHECK 2: Is the client shutting down?
                        # ============================================
                        # If dispose() or kill_agent() was called, should_stop=True
                        # We should stop waiting and return an error.
                        if self.should_stop:
                            debug_print(f"Client shutdown while waiting for response #{req_id}")
                        
                            # Put error in queue
                            response_queue.put({
                                'error': {
                                    'code': -70001,  # Internal error
                                    'message': 'Client shutdown during request'
                                }
                            })
                            # Continue to next iteration to retrieve this error
                            continue
                        
                        # Both checks passed - agent is alive and client is running
                        # Continue waiting (loop back to response_queue.get)
                        continue
                
                # ============================================================
                # RESPONSE PROCESSING - We got something from the queue
                # ============================================================
            
                debug_print(f"<<< RECEIVED RESPONSE #{req_id}")
                debug_print(json.dumps(response))
                self._spy_log('recv', 'response', response, req_id)
                
                # Check if response contains an error
                if 'error' in response:
                    error = response['error']
                    error_code = error.get('code')
                    error_message = error.get('message', str(error))
                    
                    # Log error to spy panel
                    self._spy_log('recv', 'error', response, req_id)
                    
                    # Handle authentication error (-32000)
                    if error_code == -32000:
                        # Extract additional error data
                        error_data = error.get('data')
                        data_message = ''
                        if error_data and isinstance(error_data, dict):
                            data_message = error_data.get('message', '')
                        
                        # Try authentication if not already attempted
                        if not auth_attempted and self.auth_methods:
                            debug_print(f"Authentication required (error -32000): {error_message}")
                            if data_message:
                                debug_print(f"Additional info: {data_message}")
                            try:
                                self._authenticate()
                                auth_attempted = True
                                # Retry the same request
                                continue
                            except Exception as auth_error:
                                debug_print(f"Authentication failed: {auth_error}")
                                
                                # ==============================
                                # here we manage non compliant ACP agents that do not support ACP authentication method like claude, lets show a usefull message
                                # ==============================
                                # Check if it's a manual auth requirement
                                if "Manual authentication required" in str(auth_error):
                                    # User has been notified via message queue, just propagate the error
                                    raise Exception(f"Authentication setup required. Please check the instructions in the chat window.")
                                    # ==============================
                                    # end of non ACP compliant fix
                                    # ==============================
                                else:
                                    raise Exception(f"Authentication failed: {auth_error}")
                        else:
                            # Already attempted or no auth methods available - provide helpful guidance
                            agent_name = self.agent_config.get('name', 'Agent')
                            agent_link = self.agent_config.get('link', '')
                            
                            helpful_msg = f"Authentication required for {agent_name}."
                            if data_message:
                                helpful_msg += f"\n\nDetails: {data_message}"
                            
                            if not self.auth_methods:
                                helpful_msg += "\n\nNo authentication methods available from agent."
                                helpful_msg += "\n\nTo fix this, you may need to:"
                                helpful_msg += "\n1. Login using the CLI first (run the agent command directly)"
                                helpful_msg += "\n2. Set API key in environment variables"
                                helpful_msg += "\n3. Configure API key in plugin config (Plugins > AI Agents > Open Config File)"
                                
                                if agent_link:
                                    helpful_msg += f"\n\nFor more information, visit:\n{agent_link}"
                            else:
                                helpful_msg += "\n\nAuthentication was attempted but failed."
                                helpful_msg += "\nSome authentication methods require manual setup in the terminal first."
                            
                            self.message_queue.put(('auth_required_help', helpful_msg))
                            raise Exception(helpful_msg)

                    # Map other error codes: Map standard JSON-RPC error codes to human-readable names
                    # See: https://agentclientprotocol.com/protocol/schema#errorcode
                    error_names = {
                        -32700: 'Parse error',        # Invalid JSON
                        -32600: 'Invalid request',    # Not a valid JSON-RPC request
                        -32601: 'Method not found',   # Method doesn't exist
                        -32602: 'Invalid params',     # Invalid method parameters
                        -32603: 'Agent Internal error',     # Server-side error
                        -32000: 'Authentication required',  # Need to authenticate first
                        -32002: 'Resource not found',  # Requested resource doesn't exist
    
                        -70000: 'Plugin error',  # Agent process terminated unexpectedly
                        -70001: 'Plugin error',  # Client shutdown during request
                        -70002: 'Plugin error',  # Connection closed manually
                    }
                    
                    error_name = error_names.get(error_code, f'Error {error_code}')
                            
                    # Create exception with detailed information
                    error_message_full = f"[{error_code}] {error_name}: {error_message}"

                    # Add data field if present
                    if error.get('data'):
                        data_info = error.get('data')
                        if isinstance(data_info, dict):
                            # Format dict nicely
                            data_str = ', '.join(f"{k}: {v}" for k, v in data_info.items())
                            error_message_full += f" | Data: {data_str}"
                        else:
                            error_message_full += f" | Data: {data_info}"

                    exc = Exception(error_message_full)
                    exc.error_code = error_code # Attach code for programmatic handling
                    exc.error_data = error.get('data') # Attach additional error data
                    raise exc
                
                # Success - return the result
                return response.get('result', {})
                
            finally:
                # ============================================================
                # CLEANUP - Always runs, even if exception occurred
                # ============================================================
                # Remove this request from pending list
                # Important: prevents memory leaks and ensures _cleanup_state
                # doesn't try to notify an already-completed request
                if req_id in self.pending_requests:
                    del self.pending_requests[req_id]
                     
    def _send_response(self, request_id: int, result: Any = None, error: Any = None):
        """
        Send a response to an agent request
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        response = {'jsonrpc': '2.0', 'id': request_id}
        response['error' if error else 'result'] = error if error else (result if result is not None else {})
        
        debug_print(f">>> SENDING RESPONSE to request #{request_id}")
        debug_print(json.dumps(response))
        self._spy_log('send', 'response', response, request_id)
        self.outgoing_queue.put(response)
        
    def _read_loop(self):
        """
        Read loop for stdout - processes messages from agent
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print("Read thread started")
        while not self.should_stop and self.process:
            try:
                line = self.process.stdout.readline()
                if not line:
                    break
                    
                try:
                    msg = json.loads(line)
                    debug_print(f"<<< RECEIVED MESSAGE from agent")
                    debug_print(json.dumps(msg))
                    
                    self.incoming_queue.put(msg)
                    
                    # Process message based on type
                    msg_id = msg.get('id')
                    method = msg.get('method')
                    
                    # Response to our request
                    if msg_id is not None and method is None:
                        # spy_log for Response will be logged in _send_request_and_wait_for_response when response_queue.get returns
                        if msg_id in self.pending_requests:
                            self.pending_requests[msg_id].put(msg)
                            
                    # Notification from agent
                    elif method is not None and msg_id is None:
                        self._spy_log('recv', 'notification', msg)
                        self._handle_agent_notification(msg)
                        
                    # Request from agent (e.g., permission request)
                    elif method is not None and msg_id is not None:
                        self._spy_log('recv', 'request', msg, msg_id)
                        self._handle_agent_request(msg)
                        
                except json.JSONDecodeError:
                    debug_print(f"Failed to parse JSON: {line}")
                    
            except Exception as e:
                if not self.should_stop:
                    debug_print(f'ERROR: Read error: {e}')
                break
        debug_print("Read thread stopped")
                
    def _write_loop(self):
        """
        Write loop for stdin - sends messages to agent
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print("Write thread started")
        while not self.should_stop and self.process:
            try:
                msg = self.outgoing_queue.get(timeout=1)
                json_str = json.dumps(msg) + '\n'
                self.process.stdin.write(json_str)
                self.process.stdin.flush()
            except Empty:
                continue
            except Exception as e:
                if not self.should_stop:
                    debug_print(f'ERROR: Write error: {e}')
                break
        debug_print("Write thread stopped")
                
    def _stderr_loop(self):
        """
        Read loop for stderr
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print("Stderr thread started")
        while not self.should_stop and self.process:
            try:
                line = self.process.stderr.readline()
                if not line:
                    break
                
                debug_print(f"STDERR: {line.strip()}")
                self._spy_log('stderr', 'error', line)
                
                # Queue stderr messages
                self.message_queue.put(('CLI_stderr', line))
                    
            except Exception as e:
                if not self.should_stop:
                    debug_print(f'ERROR: Stderr read error: {e}')
                break
        
        # Thread exiting - check if process died unexpectedly
        if self.process and not self.should_stop:
            exit_code = self.process.poll()
            if exit_code is not None:
                debug_print(f"Process died unexpectedly (exit code: {exit_code})")
                self.message_queue.put(('process_died', exit_code))
        
        debug_print("Stderr thread stopped")
                   
    def _handle_agent_notification(self, msg: Dict[str, Any]):
        """
        Handle notifications from agent
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        method = msg.get('method')
        params = msg.get('params', {})
        
        debug_print(f"Handling notification: {method}")
        
        if method == 'session/update':
            # Get update type
            update = params.get('update', {})
            update_type = update.get('sessionUpdate')
            
            # Handle commands update
            if update_type == 'available_commands_update':
                available_commands = update.get('availableCommands', [])
                debug_print(f"Available commands updated: {json.dumps(available_commands)}")
                self.message_queue.put(('commands_updated', available_commands))
            
            # Handle mode changes from agent
            elif update_type == 'current_mode_update':
                mode_id = update.get('modeId')
                debug_print(f"Agent changed mode to: {mode_id}")
                
                # Update local metadata
                if self.session_metadata and 'modes' in self.session_metadata:
                    modes = self.session_metadata['modes']
                    if modes:
                        modes['currentModeId'] = mode_id
                
                # Notify UI to update mode button menu
                self.message_queue.put(('mode_changed', mode_id))
            else:
                # Queue other session updates
                self.message_queue.put(('session_update', params))
                              
    def _handle_agent_request(self, msg: Dict[str, Any]):
        """
        Handle requests from agent (e.g., permission requests, tool calls)
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        method = msg.get('method')
        params = msg.get('params', {})
        req_id = msg.get('id')
        
        debug_print(f"Handling agent request: {method}")
        
        if method == 'session/request_permission':
            # Get YOLO mode setting
            yolo_enabled = getattr(self, '_yolo_mode', False)
            
            options = params.get('options', [])
            tool_call = params.get('toolCall', {})
            
            if yolo_enabled:
                # Auto-approve in YOLO mode
                # Prefer allow_always, then allow_once
                selected_option = None
                for opt in options:
                    kind = opt.get('kind', '')
                    if kind == 'allow_always':
                        selected_option = opt
                        break
                    elif kind == 'allow_once' and not selected_option:
                        selected_option = opt
                
                if selected_option:
                    debug_print(f"Auto-approving permission (YOLO mode): {selected_option.get('optionId')}")
                    self._send_response(req_id, result={
                        'outcome': {
                            'outcome': 'selected',
                            'optionId': selected_option.get('optionId')
                        }
                    })
                else:
                    # No allow option - cancel
                    debug_print("No allow option found, cancelling")
                    self._send_response(req_id, result={
                        'outcome': {
                            'outcome': 'cancelled'
                        }
                    })
            else:
                # Show permission dialog to user
                # Queue for main thread to show dialog
                response_queue = Queue()
                self.message_queue.put(('permission_request', {
                    'request_id': req_id,
                    'tool_call': tool_call,
                    'options': options,
                    'response_queue': response_queue
                }))
                
                # Wait for user response from main thread
                try:
                    user_response = response_queue.get(timeout=300)  # 5 min timeout
                    self._send_response(req_id, result=user_response)
                except Empty:
                    # Timeout - cancel
                    debug_print("Permission request timed out")
                    self._send_response(req_id, result={
                        'outcome': {
                            'outcome': 'cancelled'
                        }
                    })
        else:
            # Unknown request - send error
            debug_print(f"Unknown request method: {method}")
            print("NOTE: agent sent a non implemented method, pls open an issue in github repo", method)
            self._send_response(req_id, error={
                'code': -32601,
                'message': 'Method not found'
            })

    def cancel_operation(self):
        """
        Cancel current operation
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print("Cancelling operation")
        if self.session_id:
            self._send_request_and_wait_for_response('session/cancel', {
                'sessionId': self.session_id
            })

    def kill_agent(self, timeout=3):
        """Forcefully kill the agent process
        
        Args:
            timeout: Seconds to wait before force kill (default: 3)
        """
        debug_print("=== KILLING AGENT ===")
        self.should_stop = True
        
        if self.process:
            pid = self.process.pid
            
            if sys.platform == 'win32':
                # On Windows, use taskkill to kill process tree
                try:
                    debug_print(f"Using taskkill to kill process tree (PID {pid})")
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)], 
                                 capture_output=True, timeout=5,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
                    debug_print("Process tree killed with taskkill")
                except Exception as e:
                    debug_print(f"taskkill failed: {e}, trying process.kill()")
                    try:
                        self.process.kill()
                        self.process.wait(timeout=2)
                    except:
                        pass
            else:
                # On Unix, try graceful first
                self.process.terminate()
                try:
                    self.process.wait(timeout=timeout)
                    debug_print("Agent terminated gracefully")
                except subprocess.TimeoutExpired:
                    debug_print("Agent didn't terminate, force killing")
                    self.process.kill()
                    try:
                        self.process.wait(timeout=2)
                        debug_print("Agent killed forcefully")
                    except:
                        debug_print("WARNING: Could not confirm agent was killed")
            
            self.process = None
        
        # Close job handle on Windows (this kills all processes in the job)
        if self.job_handle and sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.kernel32.CloseHandle(self.job_handle)
                debug_print("Job handle closed")
            except:
                pass
            self.job_handle = None
        
        # Clean up state
        self._cleanup_state()

    def dispose(self):
        """Clean up resources gracefully"""
        debug_print("=== DISPOSING CLIENT ===")
        self.should_stop = True
        
        if self.process:
            pid = self.process.pid
            
            if sys.platform == 'win32':
                # On Windows, use taskkill
                try:
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(pid)],
                                 capture_output=True, timeout=5,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
                except:
                    try:
                        self.process.kill()
                    except:
                        pass
            else:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    debug_print("Process didn't terminate gracefully, forcing kill")
                    self.process.kill()
            
            self.process = None
        
        # Close job handle on Windows
        if self.job_handle and sys.platform == 'win32':
            try:
                import ctypes
                ctypes.windll.kernel32.CloseHandle(self.job_handle)
                debug_print("Job handle closed")
            except:
                pass
            self.job_handle = None
        
        # Clean up state
        self._cleanup_state()

    def _cleanup_state(self):
        """
        Clean up client state - called by dispose() and kill_agent().
        
        SAFETY VALVE - Unblocks all waiting threads:
        ============================================
        When disconnect/kill is called, there might be threads waiting in
        _send_request_and_wait_for_response for responses that will never arrive. This method
        puts error messages in ALL pending response queues to unblock them.
        
        This provides REDUNDANT PROTECTION with the periodic checks in
        _send_request_and_wait_for_response. Even if the periodic checks somehow miss the shutdown,
        this safety valve ensures no thread hangs forever.
        
        WHY BOTH MECHANISMS:
        ===================
        - Periodic checks: Proactive - detects crashes within 1 second
        - Safety valve: Reactive - ensures cleanup always unblocks waiters
        - Together: Defense in depth - robust against race conditions
        
        THREAD SAFETY:
        ==============
        This method is called from UI thread (via disconnect button) while
        _send_request_and_wait_for_response runs in worker threads. The Queue.put() operation is
        thread-safe, so this is safe.
        """
        debug_print("Cleaning up client state")
        
        # Clear session state
        self.session_id = None
        self.session_metadata = None
        self.authenticated = False
        
        # ============================================================
        # SAFETY VALVE: Notify all pending requests
        # ============================================================
        # If there are threads waiting for responses, wake them up with errors
        # so they don't hang forever after disconnect/kill
        
        debug_print(f"Notifying {len(self.pending_requests)} pending request(s)")
        
        for req_id, response_queue in list(self.pending_requests.items()):
            try:
                # Put error response in queue
                # The waiting thread will receive this and raise an exception
                response_queue.put({
                    'error': {
                        'code': -70002,  # Internal error
                        'message': 'Connection closed manually'
                    }
                })
                debug_print(f"Notified pending request #{req_id}")
            except Exception as e:
                # Queue.put() should never fail, but catch just in case
                debug_print(f"Failed to notify request #{req_id}: {e}")
        
        # Clear the pending requests dict
        self.pending_requests.clear()
        
        # ============================================================
        # Clear message queues to prevent stale messages
        # ============================================================
        # If we're shutting down, discard any unsent outgoing messages
        # and any unprocessed incoming messages
        
        while not self.outgoing_queue.empty():
            try:
                self.outgoing_queue.get_nowait()
            except Empty:
                break
        
        while not self.incoming_queue.empty():
            try:
                self.incoming_queue.get_nowait()
            except Empty:
                break
        
        # Update connection state
        self.message_queue.put(('connection_state_changed', 'disconnected'))
        debug_print("Client state cleaned up")
       
    def load_session(self, session_id: str, working_directory: str) -> None:
        """
        Load an existing session
        RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.
        """
        debug_print(f"=== LOADING SESSION {session_id} (cwd: {working_directory}) ===")
        
        if not self.process:
            raise Exception('Not connected')
        
        if not self.supports_load_session:
            raise Exception('Agent does not support loading sessions')
        
        # Send session/load request
        response = self._send_request_and_wait_for_response('session/load', {
            'sessionId': session_id,
            'cwd': working_directory,
            'mcpServers': []
        })
        
        self.session_id = session_id
        debug_print(f"Session loaded: {session_id}")
        
        return response
        
# =============================================================================
# Chat panel UI
# =============================================================================

TIMER_TAG = 'cuda_ai_agents._chat_timer'
TIMER_INTERVAL = 100  # milliseconds

class ChatPanel:
    """Chat panel for interacting with ACP agents"""
    
    # Class-level counter for unique instance IDs
    _instance_counter = 0
    
    def __init__(self, initial_agent_config=None):
        ChatPanel._instance_counter += 1
        self.instance_id = ChatPanel._instance_counter
        self.statusbar_tag = app_proc(PROC_GET_UNIQUE_TAG, '')
        
        debug_print(f"Creating ChatPanel #{self.instance_id}")
        
        # Queue for ALL thread-safe communication
        self.message_queue = Queue()
        
        # Create independent ACP client for this panel
        if initial_agent_config is None:
            initial_agent_config = get_agents_info(return_first_available=True)
        self.acp_client = ACPClient(initial_agent_config, message_queue=self.message_queue)
        
        # Backup system
        self.backup_file = None  # Current backup file path
        self.first_user_question = None  # Store first question for filename
        
        # Commands support
        self.available_commands = []  # list of available slash commands

        self.h_dlg = None
        self.h_messages = None
        self.h_input = None
        self.h_send = None
        self.h_cancel = None
        self.h_agent_button = None
        self.h_mode_button = None
        self.h_model_button = None
        self.h_predefined_model_button = None
        self.h_status_label = None
        self.h_connect = None
        self.h_disconnect = None
        self.h_yolo_check = None
        self.h_new_window = None
        self.h_spy_panel = None  # RPC spy panel
        self.h_menu_button = None  # Menu button

        # Load UI config
        self.ui_config = load_ui_config()

        self.has_session = False
        self.tools = {}
        self.agent_already_sent_first_msg = False
                
        self.is_visible = False  # Track visibility state
        self._create_ui()
        self._create_statusbar()

    def _create_ui(self):
        """Create the chat panel UI"""
        debug_print(f"Creating chat UI #{self.instance_id}")
        
        self.h_dlg = dlg_proc(0, DLG_CREATE)
        dlg_proc(self.h_dlg, DLG_PROP_SET, prop={
            'cap': _('AI Agents Chat') + f' #{self.instance_id}',
            'w': 950,
            'h': 800,
            'border': DBORDER_SIZE, # DBORDER_SIZE: Standard resizable border (on Windows: with Minimize/Maximize buttons)
            'topmost': True,
            'on_close': self._on_dialog_close,
        })

        # =============================================================================
        # 1. TOP PANEL (Fixed Container)
        # =============================================================================
        # We wrap all top controls (inputs, buttons) in a panel aligned to TOP.
        # This prevents ALIGN_CLIENT (Messages) from overlapping them.
        # Height calculation: Status(28)+5 + Agent(28)+5 + Input(140)+5 + Buttons(28)+5 ~ 245
        top_h = 250 
        n = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'panel')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=n, prop={
            'name': 'c_panel_top',
            'h': top_h,
            'align': ALIGN_TOP, # Dock to top
        })

        # Note: All controls below set 'p' (parent) to 'c_panel_top'
        
        # 1.0 +NEW BUTTON
        self.h_new_window = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_new_window, prop={
            'name': 'c_new_window',
            'p': 'c_panel_top',
            'cap': _('+'),
            'a_l': ('', '['),  # Extreme left
            'a_r': None,
            'a_t': ('', '['),
            'sp_l': 5,
            'sp_r': 5,
            'sp_t': 5,
            'sp_b': 10,
            'w': 40,
            'h': 28,
            'on_change': self._on_new_window_click,
            'hint': 'Open a new agent window',
        })

        # 1.0.5 MENU BUTTON (☰)
        self.h_menu_button = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_menu_button, prop={
            'name': 'c_menu',
            'p': 'c_panel_top',
            'cap': _('☰'),
            'a_l': ('c_new_window', ']'),
            'a_r': None,
            'a_t': ('', '['),
            'sp_l': 5,
            'sp_r': 5,
            'sp_t': 5,
            'sp_b': 10,
            'w': 40,
            'h': 28,
            'on_change': self._on_menu_button_click,
            'hint': 'Show menu options',
        })

        # 1.1 STATUS LABEL
        self.h_status_label = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'label')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_status_label, prop={
            'name': 'c_status_label',
            'p': 'c_panel_top',
            'cap': _('Status: '),
            'a_l': ('c_menu', ']'),  # After menu button
            'a_t': ('', '['),
            'sp_l': 5,
            'sp_r': 5,
            'sp_t': 8,
            'sp_b': 5,
            'h': 28,
        })
        
        self.h_status_label = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'label')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_status_label, prop={
            'name': 'c_status',
            'p': 'c_panel_top',
            'cap': _('Disconnected'),
            'font_color': status_colors.get('disconnected'),
            'a_l': ('c_status_label', ']'),
            'a_t': ('', '['),
            'sp_l': 5,
            'sp_r': 5,
            'sp_t': 8,
            'sp_b': 5,
            'h': 28,
        })
        
        # CLOSE AND CLEAN BUTTON
        self.h_close_clean = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_close_clean, prop={
            'name': 'c_close_clean',
            'p': 'c_panel_top',
            'cap': _('Close Session'),
            'a_l': None,
            'a_r': ('', ']'),
            'a_t': ('', '['),
            'sp_l': 5,
            'sp_r': 5,
            'sp_t': 5,
            'sp_b': 10,
            'w': 145,
            'h': 28,
            'on_change': self._on_close_clean_click,
            'hint': 'Full cleanup: kill agent process, remove statusbar and destroy this window',
        })

        # KILL BUTTON
        self.h_kill = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_kill, prop={
            'name': 'c_kill',
            'p': 'c_panel_top',
            'cap': _('Kill'),
            'a_l': None,
            'a_r': ('c_close_clean', '['),
            'a_t': ('', '['),
            'sp_l': 5,
            'sp_r': 5,
            'sp_t': 5,
            'sp_b': 10,
            'w': 60,
            'h': 28,
            'vis': False,  # Hidden by default (no process running)
            'on_change': self._on_kill_click,
            'hint': 'Forcefully terminate the agent but keep window open',
        })
        
        # 1.2 AGENT BUTTON MENU
        # set the agent button caption to the current agent
        current_agent = self.acp_client.agent_config
        truncated_name = self._truncate_text_for_button(current_agent["name"])

        self.h_agent_button = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_agent_button, prop={
            'name': 'c_agent_btn',
            'p': 'c_panel_top',
            # 'cap': _('Agent: (loading...)'),
            'cap': f'{truncated_name}',
            'a_l': ('', '['),
            'a_t': ('c_status', ']'),
            'sp_a': 5,
            'w': 180,
            'h': 28,
            'act': True,
            'on_change': self._on_agent_button_click,
            'hint': 'Select AI Agent',
        })

        # 1.3 PREDEFINED MODEL BUTTON MENU (for agents with supported_models in config, this agents do not follow ACP spec so we manually set its supported model in supported_models and we create here for them a special box instead of using the bellow ACP MODEL button menu to keep things seperated, once all suspported agents support sending its models as ACP spec then we can remove this)
        self.h_predefined_model_button = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_predefined_model_button, prop={
            'name': 'c_predefined_model_btn',
            'p': 'c_panel_top',
            'cap': _('Model: ...'),
            'a_l': ('c_agent_btn', ']'), 
            'a_t': ('c_status', ']'),
            'sp_a': 5,
            'w': 180,
            'h': 28,
            'vis': False,
            'act': True,
            'on_change': self._on_predefined_model_button_click,
            'hint': 'Select model (predefined)',
        })

        # 1.4 ACP MODEL BUTTON MENU  (for session-based models from ACP spec)
        self.h_model_button = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_model_button, prop={
            'name': 'c_model_btn',
            'p': 'c_panel_top',
            'cap': _('Model: ...'),
            'a_l': ('c_predefined_model_btn', ']'), 
            'a_t': ('c_status', ']'),
            'sp_a': 5,
            'w': 180,
            'h': 28,
            'vis': False,
            'act': True,
            'on_change': self._on_model_button_click,
            'hint': 'Select model',
        })

        # 1.5 MODE BUTTON MENU
        self.h_mode_button = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_mode_button, prop={
            'name': 'c_mode_btn',
            'p': 'c_panel_top',
            'cap': _('Mode: ...'),
            'a_l': ('c_model_btn', ']'),
            'a_t': ('c_status', ']'),
            'sp_a': 5,
            'w': 180,
            'h': 28,
            'vis': False,
            'act': True,
            'on_change': self._on_mode_button_click,
            'hint': 'Select mode',
        })
        
        # 1.6 DISCONNECT BUTTON
        self.h_disconnect = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_disconnect, prop={
            'name': 'c_disconnect',
            'p': 'c_panel_top',
            'cap': _('Disconnect'),
            'a_l': None,
            'a_r': ('', ']'),
            'a_t': ('c_status', ']'),
            'sp_a': 5,
            'w': 100,
            'h': 28,
            'vis': False,  # Hidden by default (not connected)
            'on_change': self._on_disconnect_click
        })

        # 1.7 CONNECT BUTTON
        self.h_connect = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_connect, prop={
            'name': 'c_connect',
            'p': 'c_panel_top',
            'cap': _('Connect'),
            'a_l': None,
            'a_r': ('c_disconnect', '['),
            'a_t': ('c_status', ']'),
            'sp_a': 5,
            'w': 80,
            'h': 28,
            'vis': True,  # Visible by default (not connected)
            'on_change': self._on_connect_click,
            'hint': 'Connect to the agent. This is not needed in general, when you send a prompt, the plugin connect automatically to the selected agent',
        })
        
        # 1.6.1 SAVE SESSION BUTTON
        self.h_save_session = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_save_session, prop={
            'name': 'c_save_session',
            'p': 'c_panel_top',
            'cap': _('Save'),
            'a_l': None,
            'a_r': ('c_connect', '['),
            'a_t': ('c_status', ']'),
            'sp_a': 5,
            'w': 60,
            'h': 28,
            'vis': False,
            'on_change': self._on_save_session_click,
            'hint': 'Save current session. Agents that support the loadSession \rcapability allow Clients to resume previous conversations. \rThis feature enables persistence across restarts and sharing\r sessions between different Client instances.'
        })
        
        # 1.6.2 LOAD SESSION BUTTON
        self.h_load_session = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_load_session, prop={
            'name': 'c_load_session',
            'p': 'c_panel_top',
            'cap': _('Load'),
            'a_l': None,
            'a_r': ('c_save_session', '['),
            'a_t': ('c_status', ']'),
            'sp_a': 5,
            'w': 60,
            'h': 28,
            'vis': True,
            'on_change': self._on_load_session_click,
            'hint': 'Load a saved session. Agents that support the loadSession \rcapability allow Clients to resume previous conversations. \rThis feature enables persistence across restarts and sharing\r sessions between different Client instances.'
        })
        
        # 1.8 INPUT BOX
        self.h_input = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'editor')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_input, prop={
            'name': 'c_input',
            'p': 'c_panel_top',
            'a_l': ('', '['), 
            'a_r': ('', ']'),
            'a_t': ('c_agent_btn', ']'), 
            'sp_a': 5,
            'h': 140,
            'hint': "I’m your AI agent. I can build full programs, refactor code, find bugs, run commands, or edit your files. Your imagination is the limit.",
            # 'texthint': "Ask me to code, debug, refactor, edit your files \ror run commands — I'm your agent and I'm ready to help",
            'texthint': "How can I help you today? I can code, debug, refactor...etc",
        })

        # Initialize the input editor
        self.ed_input = Editor(dlg_proc(self.h_dlg, DLG_CTL_HANDLE, index=self.h_input))
        self.ed_input.set_prop(PROP_GUTTER_ALL, False) # Hide gutters for input box look
        self.ed_input.set_prop(PROP_WRAP, True)
        self.ed_input.set_prop(PROP_MINIMAP, False)
        self.ed_input.set_prop(PROP_MICROMAP, False)

        # Optional: Add on_change handler to show command hints
        # TODO: is it usefull to do it?
        # self.ed_input.set_prop(PROP_ON_CHANGE, self._on_input_change)

        # 1.9 BACKUPS MENU BUTTON
        self.h_backups_menu = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_backups_menu, prop={
            'name': 'c_backups_menu',
            'p': 'c_panel_top',
            'cap': _('☰ Backups'),  # Hamburger menu icon
            'a_l': ('', '['),          # Pin to window left
            'a_r': None,
            'a_t': ('c_input', ']'),    # Pin below Input
            'sp_a': 5,
            'w': 100,
            'h': 28,
            'on_change': self._on_backups_menu_click,
            'hint': "Manage you offline session's backup files, this backups cannot be\r loaded inside the agent, so you cannot resume a session with this\r options, use the above load/save buttons to save and resume sessions.\r This serves only as a backup of your sessions. this backup files are\r created automatically, the folder content may grow with time so you\r have to clean it your self",
        })
        
        # 1.9.5 COMMANDS MENU BUTTON (next to Backups)
        self.h_commands_menu = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_commands_menu, prop={
            'name': 'c_commands_menu',
            'p': 'c_panel_top',
            'cap': _('/ Commands'),
            'a_l': ('c_backups_menu', ']'),
            'a_r': None,
            'a_t': ('c_input', ']'),
            'sp_a': 5,
            'w': 100,
            'h': 28,
            'vis': False,  # Hidden until commands are available
            'on_change': self._on_commands_menu_click
        })

        # 1.10 SEND BUTTON
        self.h_send = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_send, prop={
            'name': 'c_send',
            'p': 'c_panel_top',
            'cap': _('Send'),
            'a_l': None,               # Disable default left anchor
            'a_r': ('', ']'),          # Pin to window right
            'a_t': ('c_input', ']'),    # Pin below Input
            'sp_a': 5,
            'w': 80,
            'h': 28,
            'on_change': self._on_send_click,
            'hint': "Send your prompt. You don't need to connect first to send your prompt, it will be done automatically"
        })
        
        # 1.11 CANCEL BUTTON
        self.h_cancel = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'button')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_cancel, prop={
            'name': 'c_cancel',
            'p': 'c_panel_top',
            'cap': _('Cancel'),
            'a_l': None,
            'a_r': ('c_send', '['), # Right side attached to Send's Left
            'a_t': ('c_input', ']'),
            'sp_a': 5,
            'w': 70,
            'h': 28,
            'vis': False,  # Hidden by default (no operation running)
            'on_change': self._on_cancel_click,
            'hint': 'Cancel the current running operation',
        })

        # 1.12 YOLO CHECKBOX (moved to left of Cancel/Send buttons)
        self.h_yolo_check = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'check')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_yolo_check, prop={
            'name': 'c_yolo',
            'p': 'c_panel_top',
            'cap': _('YOLO'),
            'a_l': None,
            'a_r': ('c_cancel', '['),
            'a_t': ('c_input', ']'),
            'sp_a': 5,
            'w': 70,
            'h': 28,
            'act': True,
            'on_change': self._on_yolo_change,
            'hint': 'Enable YOLO mode: This setting allows the AI agent to execute commands and edit files without explicit user permission',
        })
        
        # =============================================================================
        # 2. SPY PANEL & SPLITTER (Bottom aligned)
        # =============================================================================
        if DEBUG_SPY:
            # 2.1 SPY PANEL (RPC Communication Log) - only if DEBUG_SPY is enabled
            self.h_spy_panel = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'editor')
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_spy_panel, prop={
                'name': 'c_spy',
                'align': ALIGN_BOTTOM,
                'h': 200,
                'sp_a': 5,
                'on_menu': self._on_spy_context_menu,
                'vis': self.ui_config.get('show_spy_panel', True),  # Load from config
                'texthint': 'You can view the Client-Agent communication here',
            })
            
            # Configure spy panel editor
            self.ed_spy = Editor(dlg_proc(self.h_dlg, DLG_CTL_HANDLE, index=self.h_spy_panel))
            self.ed_spy.set_prop(PROP_GUTTER_FOLD, False)
            self.ed_spy.set_prop(PROP_WRAP, False)
            self.ed_spy.set_prop(PROP_UNDO_LIMIT, 0)
            self.ed_spy.set_prop(PROP_GUTTER_BM, False)
            self.ed_spy.set_prop(PROP_GUTTER_NUM, False)
            self.ed_spy.set_prop(PROP_GUTTER_STATES, False)
            self.ed_spy.set_prop(PROP_RO, True)
            self.ed_spy.set_prop(PROP_MINIMAP, False)
            self.ed_spy.set_prop(PROP_LEXER_FILE, 'JSON')

            # 2.2 SPLITTER
            # Added AFTER Spy Panel so it stacks above it visually in Bottom alignment
            self.h_spy_splitter = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'splitter')
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_spy_splitter, prop={
                'name': 'sp_spy',
                'align': ALIGN_BOTTOM,
                'ex0': True, # paint border
                'ex1': True, # instant repaint
                'h': 5, # Height of splitter handle
                'sp_a': 5,
                'vis': self.ui_config.get('show_spy_panel', True),  # Load from config
            })
            
        # =============================================================================
        # 3. STDERR PANEL & SPLITTER (Bottom aligned, above spy panel if it exists)
        # =============================================================================
        # Create stderr panel (always visible, unlike spy which is DEBUG only)
        self.h_stderr_panel = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'editor')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_stderr_panel, prop={
            'name': 'c_stderr',
            'align': ALIGN_BOTTOM,
            'h': 150,
            'sp_a': 5,
            'vis': False,  # Hidden by default, shown when stderr is received
            # 'vis': self.ui_config.get('show_stderr_panel', False),  # Load from config. do not set it to True here , read bellow in the splitter
            'on_menu': self._on_stderr_context_menu,
            'texthint': 'Agents Stderr output go here',
        })
        
        # Configure stderr panel editor
        self.ed_stderr = Editor(dlg_proc(self.h_dlg, DLG_CTL_HANDLE, index=self.h_stderr_panel))
        self.ed_stderr.set_prop(PROP_GUTTER_FOLD, False)
        self.ed_stderr.set_prop(PROP_WRAP, True)
        self.ed_stderr.set_prop(PROP_UNDO_LIMIT, 0)
        self.ed_stderr.set_prop(PROP_GUTTER_BM, False)
        self.ed_stderr.set_prop(PROP_GUTTER_NUM, False)
        self.ed_stderr.set_prop(PROP_GUTTER_STATES, False)
        self.ed_stderr.set_prop(PROP_RO, True)
        self.ed_stderr.set_prop(PROP_MINIMAP, False)
        
        # Splitter for stderr panel
        self.h_stderr_splitter = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'splitter')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_stderr_splitter, prop={
            'name': 'sp_stderr',
            'align': ALIGN_BOTTOM,
            'ex0': True,
            'ex1': True,
            'h': 5,
            'sp_a': 5,
            'vis': False,  # Hidden by default
            # 'vis': self.ui_config.get('show_stderr_panel', False),  # Load from config. : TODO: BUG: if gui start with this as True i get wrong positioning of the splitter!! it works fine if i set it to True from the menu or when it is set to True when stderr messages are detected!! i tested with anchoring but was worst, align work at least, so i will set it to False and if the user want it then he can enable it from the menu, this is not a necesary panel anyway
        })

        # =============================================================================
        # 3. MESSAGES VIEW (Fill remaining space)
        # =============================================================================
        self.h_messages = dlg_proc(self.h_dlg, DLG_CTL_ADD, 'editor')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_messages, prop={
            'name': 'c_msg',
            'align': ALIGN_CLIENT, # Fill space between Top Panel and Spy/Splitter
            'sp_a': 5,
            'on_menu': self._on_results_context_menu,
        })
            
        # Configure the embedded editor for chat display
        self.ed_msg = Editor(dlg_proc(self.h_dlg, DLG_CTL_HANDLE, index=self.h_messages))

        # ed_msg.set_prop(PROP_FONT, ('default', 11))
        # self.ed_msg.set_prop(PROP_GUTTER_ALL, False)
        self.ed_msg.set_prop(PROP_GUTTER_FOLD, True)
        self.ed_msg.set_prop(PROP_WRAP, True)
        self.ed_msg.set_prop(PROP_UNDO_LIMIT, 0)
        self.ed_msg.set_prop(PROP_GUTTER_BM, False)
        self.ed_msg.set_prop(PROP_GUTTER_NUM, False) # Keep line numbers
        self.ed_msg.set_prop(PROP_GUTTER_STATES, False)
        # self.ed_msg.set_prop(PROP_HILITE_CUR_LINE, True)
        # self.ed_msg.set_prop(PROP_HILITE_CUR_LINE_IF_FOCUS, True)
        # self.ed_msg.set_prop(PROP_MODERN_SCROLLBAR, False)
        self.ed_msg.set_prop(PROP_RO, True) # Read-only by default
        self.ed_msg.set_prop(PROP_MINIMAP, True)

        # self.ed_msg.set_prop(PROP_LEXER_FILE, 'Search results')
        self.ed_msg.set_prop(PROP_LEXER_FILE, 'Markdown')
        # self.ed_msg.set_prop(PROP_LEXER_FILE, '')
                     
        # Update predefined model button menu if agent has supported_models
        self.update_predefined_model_ui()

        # Update Load button visibility based on saved sessions
        self._update_load_button_visibility()

    def _add_message(self, text, msg_type):
        """Add a message to the chat display"""
        prefix = {
            # 👤★☛☞☆☀⚙'⚐⚑-► ◄- ✘@!┌───┐└───┘│
            
            'user_question': '-► You:',
            'agent_answer': '◄- Agent:',
            'agent_answer_error': '✘ Error:',
            'system': '● System:',
        }.get(msg_type, '')
                
        now = time.strftime("%H:%M:%S")
        prefix = f'[{now}] {prefix}'
        
        if msg_type == 'new_session':
            header_text = f"# ───────── {self.acp_client.agent_config['name']} ───────── #"
            msg = f"{header_text}\n{text}"

        elif msg_type == 'user_question':
            header_text = f"─── Question ───"
            msg = f"\n{header_text}\n{prefix}\n{text}\n"
            
        elif msg_type == 'agent_answer':
            # qwen send a line word by word or 2/3 words, so we should not add \n and prefix if agent_already_sent_first_msg is true
            if not self.agent_already_sent_first_msg:
                self.agent_already_sent_first_msg = True
                msg = f'\n{prefix}\n{text}'
            else:
                msg = text
                
        elif msg_type == 'agent_answer_thought':
            # Prepend '▒' to every line in text
            lines = text.splitlines()
            processed_text = '\n'.join([f'▒ {line}' for line in lines])
            # qwen send a line word by word or 2/3 words, so we should not add \n and prefix if agent_already_sent_first_msg is true
            if not self.agent_already_sent_first_msg:
                self.agent_already_sent_first_msg = True
                msg = f'\n{prefix}\n{processed_text}\n\n'
            else:
                msg = f'\n{processed_text}\n\n'
                
        elif msg_type == 'agent_answer_error':
            msg = f'\n\n{prefix} {text}\n'
            
        elif msg_type == 'system':
            msg = f'\n\n{prefix} {text}\n'
                
        else:
            msg = f'\n\n{prefix} ▐ {text}\n'

        """Append text to the editor and scroll to bottom"""
        ed = self.ed_msg
        
        # Append text to the end
        # met1:
        # doc says: If param "y" is after the actual line-count, appends block to the end (even to final line without line-ending).
        # but it did not work! it work only with 0,0 so i have to calculate end of text to insert text correctly at the end
        # ed.insert(0,99999999,msg)
        
        # met2:
        ed.cmd(cmds.cCommand_GotoTextEnd)      # Move caret to end
        # ed.focus()
        ed.set_prop(PROP_RO, False)
        ed.cmd(cmds.cCommand_TextInsert, text=msg)
        ed.set_prop(PROP_RO, True)
                
        # Ensure visible (scroll to end)
        ed.cmd(cmds.cCommand_ScrollToEnd)

        # Backup handling
        if msg_type == 'user_question':
            # Store first question for filename
            if not self.first_user_question:
                self.first_user_question = text[:100]  # Store first 100 chars
                
            # Create backup file on first question
            if not self.backup_file:
                self.backup_file = self._create_backup_file(self.first_user_question)
            
            # Save backup after user question
            self._save_backup()
            
        elif msg_type in ('agent_answer', 'agent_answer_thought', 'system'):
            # Save backup after agent responses
            if self.backup_file:
                self._save_backup()

    def on_timer(self):
        """Timer callback - processes messages from worker thread (runs on main thread so we can run cudatext api here)"""
        if not self.message_queue:
            return
        
        try:
            while True:
                try:
                    msg_type, msg_data = self.message_queue.get_nowait()
                    
                    if msg_type == 'process_died':
                        exit_code = msg_data
                        self._add_message(_(f'⚠ Agent process terminated unexpectedly (exit code: {exit_code})'), 'system')
                        
                        # Clean up and update state
                        self.acp_client.process = None
                        self.acp_client._cleanup_state()
                        
                        # Hide kill button since process is gone
                        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_kill, prop={'vis': False})

                    elif msg_type == 'auth_required_help':
                        # Show detailed authentication help message
                        help_text = msg_data
                        self._add_message(help_text, 'system')
                        
                    elif msg_type == 'auth_manual_setup_required':
                        # Handle authentication methods that require manual setup, this is mostly a non ACP compliant agent like claude
                        req_data = msg_data
                        message = req_data['message']
                        method_name = req_data['method_name']
                        description = req_data['description']
                        
                        # Show detailed message in chat
                        self._add_message(message, 'system')
                        
                        # Also show a dialog for immediate attention
                        msg_box(message, f'Manual Authentication Required: {method_name}', MB_OK + MB_ICONWARNING)
    
                    elif msg_type == 'commands_updated':
                        # Update available commands
                        commands = msg_data
                        self.available_commands = commands
                        debug_print(f"Updated available commands: {len(commands)} commands")
                        
                        # Show/hide commands button
                        has_commands = len(commands) > 0
                        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_commands_menu, 
                                prop={'vis': has_commands})
                        
                        if has_commands:
                            # Log available commands
                            cmd_names = [cmd.get('name', '') for cmd in commands]
                            self._add_message(_(f'This agent accepts commands: {", ".join(["/" + n for n in cmd_names])}'), 'system')
                            
                    elif msg_type == 'mode_changed':
                        # Agent notified us of mode change
                        mode_id = msg_data
                        self._sync_mode_from_agent(mode_id)
                                                    
                    elif msg_type == 'session_metadata':
                        # Metadata received/updated - refresh all UI
                        debug_print("Updating session metadata UI")
                        self.update_mode_ui()
                        self.update_model_ui()
                        self.update_predefined_model_ui()
                        
                        # Apply pending settings AFTER metadata is loaded
                        # this apply all saved user preferences after connection.
                        self._apply_pending_settings_after_connection()
                        
                        # Show/hide save/load buttons based on capabilities
                        has_session = (self.acp_client.session_id is not None)
                        
                        # Save button: only show if we have an active session
                        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_save_session, 
                                prop={'vis': has_session})
                        
                        # Load button: show if agent has saved sessions (already set by _update_load_button_visibility)
                        # But update if we need to hide it because agent doesn't support loading
                        if self.acp_client.state == 'connected' and not self.acp_client.supports_load_session:
                            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_load_session, 
                                    prop={'vis': False})
                        else:
                            # Keep current visibility (based on saved sessions)
                            self._update_load_button_visibility()
                    
                    elif msg_type == 'permission_request':
                        # Handle permission request in main thread
                        req_data = msg_data
                        request_id = req_data['request_id']
                        tool_call = req_data['tool_call']
                        options = req_data['options']
                        response_queue = req_data['response_queue']
                        
                        # Build dialog
                        tool_title = tool_call.get('title', 'Permission Request')
                        
                        # Build menu items with visual indicators
                        menu_items = []
                        for opt in options:
                            name = opt.get('name', opt.get('optionId', 'Unknown'))
                            kind = opt.get('kind', '')
                            
                            # Add visual indicators based on kind
                            if kind == 'allow_always':
                                icon = '✓✓'
                            elif kind == 'allow_once':
                                icon = '✓'
                            elif kind == 'reject_always':
                                icon = '✘✘'
                            elif kind == 'reject_once':
                                icon = '✘'
                            else:
                                icon = '•'
                            
                            menu_items.append(f'{icon} {name}')
                        
                        # Show dialog
                        selected_idx = dlg_menu(DMENU_LIST, '\n'.join(menu_items), 
                                              caption=f'Agent Permission: {tool_title}')
                        
                        # Send response back to worker thread
                        if selected_idx is not None and 0 <= selected_idx < len(options):
                            selected_option = options[selected_idx]
                            response_queue.put({
                                'outcome': {
                                    'outcome': 'selected',
                                    'optionId': selected_option.get('optionId')
                                }
                            })
                        else:
                            # User cancelled
                            response_queue.put({
                                'outcome': {
                                    'outcome': 'cancelled'
                                }
                            })

                    elif msg_type == 'auth_method_selection':
                        # Handle authentication method selection in main thread
                        req_data = msg_data
                        methods = req_data['methods']
                        menu_items = req_data['menu_items']
                        response_queue = req_data['response_queue']
                        
                        # Show dialog to user
                        selected_idx = dlg_menu(DMENU_LIST_ALT, '\n'.join(menu_items), 
                                              caption='Select Authentication Method')
                        
                        # Send response back to worker thread
                        if selected_idx is not None and 0 <= selected_idx < len(methods):
                            selected_method = methods[selected_idx]
                            response_queue.put(selected_method['id'])
                        else:
                            # User cancelled
                            response_queue.put(None)

                    elif msg_type == 'connection_state_changed':
                        # Handle connection state change and update status label and button
                        state = msg_data
                        old_state = self.acp_client.state
                        if old_state != state:
                            debug_print(f"State changed: {old_state} -> {state}")
                            self.acp_client.state = state
                            
                            # Get status text            
                            status_text = status_texts.get(state, state)
                            color = status_colors.get(state, 0x808080)

                            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_status_label, prop={
                                'cap': status_text,
                                'font_color': color
                            })
                            
                            # Show/hide connect/disconnect buttons based on state
                            is_connected = (state == 'connected')
                            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_connect, prop={'vis': not is_connected})
                            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_disconnect, prop={'vis': is_connected})
                            
                            # Show/hide kill button based on whether process is running
                            is_connecting = (state == 'connecting')
                            has_process = (self.acp_client.process is not None)
                            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_kill, prop={'vis': has_process or is_connecting})
                            
                            # Update statusbars
                            self._update_statusbar()
                            
                            # Update main statusbar
                            global _command_instance
                            if _command_instance:
                                _command_instance._update_main_statusbar()
                                
                    elif msg_type == 'user_question':
                        self._add_message(msg_data, 'user_question')
                    elif msg_type == 'new_session':
                        # this is a new session
                        self._add_message(msg_data, 'new_session')

                    elif msg_type == 'session_update':
                        # Handle session updates from agent
                        params = msg_data
                        update_type = params.get('update', {}).get('sessionUpdate')
                        debug_print(f"Session update received: {update_type}")
                        
                        # Add handlers for session replay during load
                        if update_type == 'user_message_chunk':
                            # Replay user message from history
                            content = params.get('update', {}).get('content', {})
                            if content.get('type') == 'text':
                                chunk = content.get('text', '')
                                self._add_message(f'{chunk}', 'user_question')
                        
                        elif update_type == 'agent_message_chunk':
                            # Append answer chunk to the messages box
                            content = params.get('update', {}).get('content', {})
                            if content.get('type') == 'text':
                                chunk = content.get('text', '')
                                self._add_message(f'{chunk}', 'agent_answer')

                        elif update_type == 'agent_thought_chunk':
                            # Append answer thinking chunk to the messages box
                            content = params.get('update', {}).get('content', {})
                            if content.get('type') == 'text':
                                chunk = content.get('text', '')
                                self._add_message(f'Thinking: {chunk}', 'agent_answer_thought')
                        
                        elif update_type == 'tool_call':
                            # Tool execution started
                            tool_name = params.get('update', {}).get('title', 'Tool')
                            tool_id = params.get('update', {}).get('toolCallId')
                            debug_print(f"Tool call: {tool_name} (ID: {tool_id})")
                            self.tools[tool_id] = {
                                'name': tool_name,
                                'status': 'running'
                            }                            
                            self._add_message(_(f'→ Executing tool: {tool_name}'), 'system')
                        
                        elif update_type == 'tool_call_update':
                            tool_id = params.get('update', {}).get('toolCallId')
                            if tool_id in self.tools:
                                status = params.get('update', {}).get('status')
                                debug_print(f"Tool {tool_id} status: {status}")
                                self.tools[tool_id]['status'] = status
                                
                                # Get content if available (error messages, etc.)
                                content_list = params.get('update', {}).get('content', [])
                                content_text = ''
                                for content_item in content_list:
                                    if content_item.get('type') == 'content':
                                        inner_content = content_item.get('content', {})
                                        if inner_content.get('type') == 'text':
                                            content_text = inner_content.get('text', '')
                                
                                # Tool execution status update
                                tool_name = self.tools[tool_id]['name']                        
                                if status == 'completed':
                                    self._add_message(_(f'✓ Tool completed: {tool_name}'), 'system')
                                elif status == 'failed':
                                    msg = _(f'✘ Tool failed: {tool_name}')
                                    if content_text:
                                        msg += f'\n  {content_text}'
                                    self._add_message(msg, 'system')
                                elif status == 'denied':
                                    msg = _(f'✘ Tool denied: {tool_name}')
                                    if content_text:
                                        msg += f'\n  {content_text}'
                                    self._add_message(msg, 'system')
                            
                    elif msg_type == 'CLI_stderr':
                        # Handle stderr output from agent - show in dedicated panel
                        stderr_line = msg_data
                        
                        # Filter specific Claude startup message, this is not an error, claude print to stderr a useless message which opens stderr panel, lets filter it
                        if "Spawning Claude Code:" in stderr_line and self.acp_client.agent_config['id'] == 'claude-code':
                            continue
        
                        # Show stderr panel if hidden (first stderr message)
                        if hasattr(self, 'h_stderr_panel'):
                            is_visible = dlg_proc(self.h_dlg, DLG_CTL_PROP_GET, index=self.h_stderr_panel)['vis']
                            if not is_visible:
                                dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_stderr_panel, prop={'vis': True})
                                dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_stderr_splitter, prop={'vis': True})
                            
                            # Append to stderr panel
                            ed = self.ed_stderr
                            ed.cmd(cmds.cCommand_GotoTextEnd)
                            ed.set_prop(PROP_RO, False)
                            ed.cmd(cmds.cCommand_TextInsert, text=stderr_line)
                            ed.set_prop(PROP_RO, True)
                            ed.cmd(cmds.cCommand_ScrollToEnd)
        
                    elif msg_type == 'agent_answer_error':
                        self._add_message(f'{msg_data}', 'agent_answer_error')
                        
                        # Hide cancel button on error
                        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_cancel, prop={'vis': False})
                        
                    elif msg_type == 'agent_answer_start':
                        # TODO: Could show "thinking" indicator
                        # Add agent_answer prefix on start for better UX
                        self._add_message('', 'agent_answer')  # Adds '\n◄- Agent:\n'
                        
                        # Show cancel button when operation starts
                        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_cancel, prop={'vis': True})
                        
                    elif msg_type == 'agent_answer_end':
                        # Handle stop reason from agent response
                        stop_reason = msg_data
                        if stop_reason == 'end_turn':
                            self._add_message(_('✓ Agent finished response'), 'system')
                        elif stop_reason == 'cancelled':
                            self._add_message(_('✘ Operation cancelled by user'), 'system')
                        elif stop_reason == 'max_turns':
                            self._add_message(_('✘ Max turns reached'), 'system')
                        elif stop_reason == 'error':
                            self._add_message(_('✘ Agent encountered an error'), 'system')
                        else:
                            self._add_message(_(f'■ Agent stopped: {stop_reason}'), 'system')

                        # Hide cancel button when operation ends
                        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_cancel, prop={'vis': False})
                        
                    elif msg_type == 'spy':
                        # Handle spy log messages - append to spy panel
                        if DEBUG_SPY and hasattr(self, 'ed_spy'):
                            try:
                                ed = self.ed_spy
                                ed.cmd(cmds.cCommand_GotoTextEnd)
                                ed.set_prop(PROP_RO, False)
                                ed.cmd(cmds.cCommand_TextInsert, text=msg_data)
                                ed.set_prop(PROP_RO, True)
                                ed.cmd(cmds.cCommand_ScrollToEnd)
                            except Exception as e:
                                debug_print(f"Error appending to spy panel: {e}")
                    else:
                        self._add_message(_(f'■■■ thissouldnothappen: {msg_data}'), 'thissouldnothappen')
  
                except Empty:
                    break
        except Exception as e:
            debug_print(f"ERROR: AI Agents: in on_timer for panel #{self.instance_id}: {e}")

    # ===statusbar==========================================================================
    def ___statusbar____________________________________________():
        pass
        
    def _create_statusbar(self):
        """Create statusbar cell for this panel"""
        # Find position after main AI Agents cell
        main_cell_index = statusbar_proc(BAR_H, STATUSBAR_FIND_CELL, value=CELL_TAG)
        if main_cell_index is None:
            return
        
        # Add cell after main cell
        statusbar_proc(BAR_H, STATUSBAR_ADD_CELL, index=main_cell_index + 1, tag=self.statusbar_tag)
        
        # Get colors from main cell
        color_back = statusbar_proc(BAR_H, STATUSBAR_GET_CELL_COLOR_BACK, tag=CELL_TAG)
        color_font = statusbar_proc(BAR_H, STATUSBAR_GET_CELL_COLOR_FONT, tag=CELL_TAG)
        
        statusbar_proc(BAR_H, STATUSBAR_SET_CELL_COLOR_BACK, tag=self.statusbar_tag, value=color_back)
        statusbar_proc(BAR_H, STATUSBAR_SET_CELL_COLOR_FONT, tag=self.statusbar_tag, value=color_font)
        statusbar_proc(BAR_H, STATUSBAR_SET_CELL_ALIGN, tag=self.statusbar_tag, value='C')
        statusbar_proc(BAR_H, STATUSBAR_SET_CELL_AUTOSIZE, tag=self.statusbar_tag, value=True)
        statusbar_proc(BAR_H, STATUSBAR_SET_CELL_CALLBACK, tag=self.statusbar_tag, 
                      value=f'module=cuda_ai_agents;cmd=callback_panel_statusbar_click;info={self.instance_id};')
        
        self._update_statusbar()
        
    def _update_statusbar(self):
        """Update this panel's statusbar cell"""        
        state = self.acp_client.state
        icon = status_icons.get(state, '◯')
        color = status_colors.get(state, 0x808080)
        agent_name = self.acp_client.agent_config['name']
        
        text = f'{agent_name} #{self.instance_id} {icon}'
        
        statusbar_proc(BAR_H, STATUSBAR_SET_CELL_TEXT, tag=self.statusbar_tag, value=text)
        statusbar_proc(BAR_H, STATUSBAR_SET_CELL_COLOR_FONT, tag=self.statusbar_tag, value=color)
        statusbar_proc(BAR_H, STATUSBAR_SET_CELL_AUTOSIZE, tag=self.statusbar_tag, value=True)
    
    def _remove_statusbar(self):
        """Remove this panel's statusbar cell"""
        index = statusbar_proc(BAR_H, STATUSBAR_FIND_CELL, value=self.statusbar_tag)
        if index is not None:
            statusbar_proc(BAR_H, STATUSBAR_DELETE_CELL, index=index)
        
    # ===msg_panel==========================================================================
    def ___msg_panel____________________________________________():
        pass
        
    def _on_results_context_menu(self, id_dlg, id_ctl, data='', info=''):
        """Handle context menu for messages editor - create custom menu with fold/unfold commands.
        i could not find a way to edit the default menu, i dont know how to get the handle of the menu, so here i recreate the default menu and add my wanted commands
        """
        
        ed = self.ed_msg
        # if not ed:
            # return False  # Allow default menu
        
        # Create a custom menu
        h_menu = menu_proc(0, MENU_CREATE)

        # Add some useful default editor commands
        menu_proc(h_menu, MENU_ADD, caption='Copy', command=lambda: ed.cmd(cmds.cCommand_ClipboardCopy))
        menu_proc(h_menu, MENU_ADD, caption='Select All', command=lambda: ed.cmd(cmds.cCommand_SelectAll))
        
        menu_proc(h_menu, MENU_ADD, caption='-')  # Separator
        
        # Add our custom menu items
        # menu_proc(h_menu, MENU_ADD, caption='Fold', command=lambda: ed.cmd(cmds.cCommand_FoldAll))
        menu_proc(h_menu, MENU_ADD, caption='Fold', command=lambda: ed.cmd(cmds.cCommand_FoldLevel2))
        menu_proc(h_menu, MENU_ADD, caption='Unfold', command=lambda: ed.cmd(cmds.cCommand_UnfoldAll))
        
        menu_proc(h_menu, MENU_ADD, caption='Toggle word wrap', command=lambda: ed.cmd(cmds.cCommand_ToggleWordWrap))
        # menu_proc(h_menu, MENU_ADD, caption='Toggle word wrap2', command=lambda: ed.cmd(cmds.cCommand_ToggleWordWrapAlt))
        
        menu_proc(h_menu, MENU_ADD, caption='-')  # Separator

        # menu_proc(h_menu, MENU_ADD, caption='Clear', command=lambda *args,**vargs: self.clear_messages(id_dlg, id_ctl, data, info))
        # menu_proc(h_menu, MENU_ADD, caption='Clear', command=lambda: self.clear_messages(id_dlg, id_ctl, data, info))
        menu_proc(h_menu, MENU_ADD, caption='Clear', command=lambda: self.clear_messages())
        
        # Show the menu
        menu_proc(h_menu, MENU_SHOW)
        
        # Return False to prevent the default context menu from showing
        return False

    def clear_messages(self):
        """Clear the entire message display"""
        ed = self.ed_msg
        ed.set_prop(PROP_RO, False)
        ed.set_text_all('')
        ed.set_prop(PROP_RO, True)

    # ===spy panels==========================================================================
    def ___spy_panel____________________________________________():
        pass
        
    def _on_spy_context_menu(self, id_dlg, id_ctl, data='', info=''):
        """Handle context menu for spy panel - create custom menu with useful commands"""
        
        ed = self.ed_spy
        
        # Create a custom menu
        h_menu = menu_proc(0, MENU_CREATE)
        
        # Add useful editor commands
        menu_proc(h_menu, MENU_ADD, caption='Copy', command=lambda: ed.cmd(cmds.cCommand_ClipboardCopy))
        menu_proc(h_menu, MENU_ADD, caption='Select All', command=lambda: ed.cmd(cmds.cCommand_SelectAll))
        
        menu_proc(h_menu, MENU_ADD, caption='-')  # Separator
        
        # Add spy-specific commands
        menu_proc(h_menu, MENU_ADD, caption='Clear Spy Log', command=lambda: self.clear_spy_log())
        menu_proc(h_menu, MENU_ADD, caption='Toggle word wrap', command=lambda: ed.cmd(cmds.cCommand_ToggleWordWrap))
        
        # Show the menu
        menu_proc(h_menu, MENU_SHOW)
        
        # Return False to prevent the default context menu from showing
        return False

    def clear_spy_log(self):
        """Clear the spy panel"""
        if hasattr(self, 'ed_spy'):
            ed = self.ed_spy
            ed.set_prop(PROP_RO, False)
            ed.set_text_all('')
            ed.set_prop(PROP_RO, True)
        
    # =============================================================================
    # stderr panel
    # =============================================================================
    def ___stderr_panel____________________________________________():
        pass
        
    def _on_stderr_context_menu(self, id_dlg, id_ctl, data='', info=''):
        """Handle context menu for stderr panel"""
        
        ed = self.ed_stderr
        
        # Create a custom menu
        h_menu = menu_proc(0, MENU_CREATE)
        
        # Add useful editor commands
        menu_proc(h_menu, MENU_ADD, caption='Copy', command=lambda: ed.cmd(cmds.cCommand_ClipboardCopy))
        menu_proc(h_menu, MENU_ADD, caption='Select All', command=lambda: ed.cmd(cmds.cCommand_SelectAll))
        
        menu_proc(h_menu, MENU_ADD, caption='-')  # Separator
        
        # Add stderr-specific commands
        menu_proc(h_menu, MENU_ADD, caption='Clear Stderr Log', command=lambda: self.clear_stderr_log())
        menu_proc(h_menu, MENU_ADD, caption='Hide Stderr Panel', command=lambda: self.hide_stderr_panel())
        menu_proc(h_menu, MENU_ADD, caption='Toggle word wrap', command=lambda: ed.cmd(cmds.cCommand_ToggleWordWrap))
        
        # Show the menu
        menu_proc(h_menu, MENU_SHOW)
        
        return False

    def clear_stderr_log(self):
        """Clear the stderr panel"""
        if hasattr(self, 'ed_stderr'):
            ed = self.ed_stderr
            ed.set_prop(PROP_RO, False)
            ed.set_text_all('')
            ed.set_prop(PROP_RO, True)
            
    def hide_stderr_panel(self):
        """Hide the stderr panel"""
        if hasattr(self, 'h_stderr_panel'):
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_stderr_panel, prop={'vis': False})
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_stderr_splitter, prop={'vis': False})
            
    # =============================================================================
    # generic menu to authenticate and show/hide spy panel and stderr panel
    # =============================================================================
    def __menu____________________________________________():
        pass
        
    def _on_menu_button_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle menu button click - show dropdown menu"""
        debug_print("Menu button clicked")
        
        # Create popup menu
        h_menu = menu_proc(0, MENU_CREATE)
        
        # Add Authenticate Manually option
        menu_proc(h_menu, MENU_ADD, 
                 caption=_('Authenticate Manually'), 
                 command=lambda: self._authenticate_manually())
        
        menu_proc(h_menu, MENU_ADD, caption='-')  # Separator
        
        # Add Spy Panel toggle (only if DEBUG_SPY is enabled)
        if DEBUG_SPY and hasattr(self, 'h_spy_panel'):
            spy_visible = dlg_proc(self.h_dlg, DLG_CTL_PROP_GET, index=self.h_spy_panel)['vis']
            spy_caption = _('Hide Spy Panel') if spy_visible else _('Show Spy Panel')
            menu_proc(h_menu, MENU_ADD, 
                     caption=spy_caption, 
                     command=lambda: self._toggle_spy_panel())
        
        # Add Stderr Panel toggle
        if hasattr(self, 'h_stderr_panel'):
            stderr_visible = dlg_proc(self.h_dlg, DLG_CTL_PROP_GET, index=self.h_stderr_panel)['vis']
            stderr_caption = _('Hide Stderr Panel') if stderr_visible else _('Show Stderr Panel')
            menu_proc(h_menu, MENU_ADD, 
                     caption=stderr_caption, 
                     command=lambda: self._toggle_stderr_panel())
        
        # Show the menu
        menu_proc(h_menu, MENU_SHOW)

    def _authenticate_manually(self):
        """Launch manual authentication in a background thread"""
        debug_print("Manual authentication requested")
        
        if self.acp_client.state != 'connected':
            msg_status(_('Not connected. Please connect first.'))
            return
        
        if not self.acp_client.auth_methods:
            msg_status(_('No authentication methods available'))
            return
        
        def auth_thread():
            '''RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.'''
            try:
                self.acp_client._authenticate()
                self.message_queue.put(('system', '✓ Authentication completed'))
            except Exception as e:
                debug_print(f"Manual authentication failed: {e}")
                self.message_queue.put(('agent_answer_error', f'Authentication failed: {str(e)}'))
        
        threading.Thread(target=auth_thread, daemon=True).start()
        
        # Ensure timer is running
        global _command_instance
        if _command_instance:
            _command_instance._ensure_timer_running()

    def _toggle_spy_panel(self):
        """Toggle spy panel visibility and save to config"""
        if not DEBUG_SPY or not hasattr(self, 'h_spy_panel'):
            return
        
        # Get current visibility
        is_visible = dlg_proc(self.h_dlg, DLG_CTL_PROP_GET, index=self.h_spy_panel)['vis']
        new_visibility = not is_visible
        
        # Update UI
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_spy_panel, 
                prop={'vis': new_visibility})
        
        if hasattr(self, 'h_spy_splitter'):
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_spy_splitter, 
                    prop={'vis': new_visibility})
        
        # Update config
        self.ui_config['show_spy_panel'] = new_visibility
        save_ui_config(show_spy_panel=new_visibility)
        
        status = 'shown' if new_visibility else 'hidden'
        msg_status(_(f'Spy panel {status}'))
        debug_print(f"Spy panel {status}")

    def _toggle_stderr_panel(self):
        """Toggle stderr panel visibility and save to config"""
        if not hasattr(self, 'h_stderr_panel'):
            return
        
        # Get current visibility
        is_visible = dlg_proc(self.h_dlg, DLG_CTL_PROP_GET, index=self.h_stderr_panel)['vis']
        new_visibility = not is_visible
        
        # Update UI
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_stderr_panel, 
                prop={'vis': new_visibility})
        
        if hasattr(self, 'h_stderr_splitter'):
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_stderr_splitter, 
                    prop={'vis': new_visibility})
        
        # Update config
        self.ui_config['show_stderr_panel'] = new_visibility
        save_ui_config(show_stderr_panel=new_visibility)
        
        status = 'shown' if new_visibility else 'hidden'
        msg_status(_(f'Stderr panel {status}'))
        debug_print(f"Stderr panel {status}")
        
    # ===buttons==========================================================================
    def ___buttons____________________________________________():
        pass
        
    def _on_yolo_change(self, id_dlg, id_ctl, data='', info=''):
        """Handle YOLO checkbox change"""
        yolo_val = dlg_proc(self.h_dlg, DLG_CTL_PROP_GET, index=self.h_yolo_check)['val']
        yolo_enabled = str_to_bool(yolo_val)
        
        # Save user's YOLO preference
        update_user_choice(None, 'yolo_enabled', yolo_enabled)
        debug_print(f"YOLO mode {'enabled' if yolo_enabled else 'disabled'}")
        
    def _on_close_clean_click(self, id_dlg, id_ctl, data='', info=''):
        """Full cleanup: kill process, remove statusbar, destroy UI, unregister"""
        debug_print(f"Full cleanup initiated for panel #{self.instance_id}")
        
        # 0. Finalize backup before cleanup
        self._finalize_backup()
        
        # 1. Kill the agent process
        if self.acp_client:
            self.acp_client.kill_agent()
        
        # 2. COMPREHENSIVE RESET (cleanup state before destroying UI)
        self._reset_all_ui_and_state()
            
        # 3. Remove this panel's statusbar cell
        self._remove_statusbar()
        
        # 4. Destroy the dialog window entirely
        if self.h_dlg:
            dlg_proc(self.h_dlg, DLG_FREE)
            self.h_dlg = None
            
        # 5. Unregister from Command tracking: Remove from global tracking in Command class
        global _command_instance
        if _command_instance and self in _command_instance.chat_panels:
            _command_instance.chat_panels.remove(self)
            debug_print(f"Removed panel #{self.instance_id} from tracking")
            
            # Update the main AI statusbar (it needs to recalculate counts)
            _command_instance._update_main_statusbar()
            
            # If no panels are left, stop the shared timer to save CPU
            if not _command_instance.chat_panels:
                _command_instance._stop_timer()
        
        msg_status(_(f"AI Agent #{self.instance_id} closed and cleaned up."))

    def _on_connect_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle connect button click - also creates session to get metadata"""
        debug_print("Connect button clicked")
        if self.acp_client.state != 'connected':
            def connect_thread():
                '''RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.'''
                try:
                    # Apply predefined model to args BEFORE connecting
                    self._apply_predefined_model_to_args()
                    
                    # Connect to agent (gets modes from initialize)
                    self.acp_client.establish_agent_connection()
                    msg_status(_('Connected to AI agent'))
                    
                    # Queue metadata update from initialize response
                    self.message_queue.put(('session_metadata', None))
                    
                    # Create session to get models (and potentially more modes info)
                    debug_print("Creating session to retrieve full metadata...")
                    self.message_queue.put(('new_session', ''))
                    cwd = os.getcwd()
                    if ed.get_filename():
                        cwd = os.path.dirname(ed.get_filename())
                    self.acp_client.new_session(cwd)
                    self.has_session = True
                    
                    # Queue another metadata update after session creation
                    self.message_queue.put(('session_metadata', None))
                    
                except Exception as e:
                    debug_print(f"Connection error: {e}")
                    msg_status(_('Failed to connect: ') + str(e))
                    self.message_queue.put(('agent_answer_error', f'Connection failed: {str(e)}'))
                    if DEBUG:
                        import traceback
                        traceback.print_exc()

            threading.Thread(target=connect_thread, daemon=True).start()
            
            # Ensure timer is running
            global _command_instance
            if _command_instance:
                _command_instance._ensure_timer_running()
        else:
            msg_status(_('Already connected'))

    def _on_disconnect_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle disconnect button click - graceful disconnect, keep window open"""
        debug_print("Disconnect button clicked")
        if self.acp_client.state == 'connected':
            self.acp_client.dispose()
            
            # COMPREHENSIVE RESET
            self._reset_all_ui_and_state()
            
            msg_status(_('Disconnected from AI agent'))
        else:
            msg_status(_('Not connected'))

    def _on_new_window_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle +New button click - open a new agent window"""
        debug_print("+New button clicked")
        global _command_instance
        if _command_instance:
            _command_instance.plugin_start_new_chat_window()

    def _on_cancel_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle cancel button click - cancel current operation"""
        debug_print("Cancel button clicked")
        if self.acp_client.session_id:
            self._add_message(_('Cancelling operation...'), 'system')
            
            # Hide cancel button immediately
            # dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_cancel, prop={'vis': False})
            
            def cancel_thread():
                '''RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.'''
                try:
                    self.acp_client.cancel_operation()
                except Exception as e:
                    debug_print(f"Cancel error: {e}")
                    self.message_queue.put(('agent_answer_error', f'Cancel failed: {str(e)}'))
            
            threading.Thread(target=cancel_thread, daemon=True).start()
        else:
            msg_status(_('No active operation to cancel'))

    def _on_kill_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle kill button click - forcefully terminate agent but keep window open"""
        debug_print("Kill button clicked")
        
        self.acp_client.kill_agent()
        
        # COMPREHENSIVE RESET
        self._reset_all_ui_and_state()
        
        self._add_message(_('Agent process terminated. Reconnect to continue.'), 'system')
        
    def _on_dialog_close(self, id_dlg, id_ctl, data='', info=''):
        """Handle dialog close (X button) - just hide window, keep agent alive"""
        debug_print(f"Dialog #{self.instance_id} closing (hiding, agent stays alive)")

        # Save backup before hiding
        self._finalize_backup()

        self.is_visible = False
        
        # Just hide the dialog, don't kill agent
        # Do NOT call dlg_proc(..., DLG_HIDE) here; it causes RecursionError. Dialog framework handles the hiding
        # dlg_proc(self.h_dlg, DLG_HIDE)
        
        # Don't remove from tracking - keep panel in list so it can be reopened

    def show(self):
        """Show the chat panel and update state"""
        debug_print(f"Showing chat panel #{self.instance_id}")
        dlg_proc(self.h_dlg, DLG_SHOW_NONMODAL)
        dlg_proc(self.h_dlg, DLG_FOCUS)
        self.is_visible = True

    def hide(self):
        """Hide the chat panel and update state"""
        debug_print(f"Hiding chat panel #{self.instance_id}")
        dlg_proc(self.h_dlg, DLG_HIDE)
        self.is_visible = False

    def toggle(self):
        """Toggle visibility based on current state"""
        if self.is_visible:
            self.hide()
        else:
            self.show()
    
    def _on_send_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle send button click"""
        text = self.ed_input.get_text_all()
        
        if text.strip():
            self._handle_and_send_user_message(text.strip())
            
            # Clear the input editor after sending
            self.ed_input.set_text_all('')
            self.ed_input.focus()
            
    def _handle_and_send_user_message(self, text):
        """Send message to agent - runs in background thread"""
        
        # Update YOLO mode in ACP client before sending
        yolo_val = dlg_proc(self.h_dlg, DLG_CTL_PROP_GET, index=self.h_yolo_check)['val']
        self.acp_client._yolo_mode = str_to_bool(yolo_val)  # Convert 1/0 to True/False
        
        def send_thread():
            '''RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.'''
            try:
                # Connect if not connected
                if self.acp_client.state != 'connected':
                    debug_print("Not connected, connecting...")
                    
                    # Apply predefined model to args BEFORE connecting
                    self._apply_predefined_model_to_args()
                    
                    self.acp_client.establish_agent_connection()
                    
                # Create session if needed
                if not self.has_session:
                    debug_print("No session, creating new session...")
                    self.message_queue.put(('new_session', ''))
                    cwd = os.getcwd()
                    if ed.get_filename():
                        cwd = os.path.dirname(ed.get_filename())
                    self.acp_client.new_session(cwd)
                    self.has_session = True
                    
                    # Queue metadata update immediately after session creation
                    self.message_queue.put(('session_metadata', None))
                        
                # Send message
                debug_print(f"User sending message: {text[:50]}...")
                self.message_queue.put(('user_question', text))
                self.message_queue.put(('agent_answer_start', None))
                response = self.acp_client.send_prompt_to_agent(text)
                
                # Handle stop reason
                stop_reason = response.get('stopReason', 'unknown')
                debug_print(f"Agent stopped with reason: {stop_reason}")
                self.message_queue.put(('agent_answer_end', stop_reason))
                    
            except Exception as e:
                debug_print(f"ERROR: in send_thread: {e}")
                self.message_queue.put(('agent_answer_error', str(e)))
                if DEBUG:
                    import traceback
                    traceback.print_exc()
                
        # Start worker thread
        threading.Thread(target=send_thread, daemon=True).start()
        
        # Start timer to check queue: Ensure global timer is running (Command class manages this)
        global _command_instance
        if _command_instance:
            _command_instance._ensure_timer_running()
            
    # =============================================================================
    # online session save and load
    # =============================================================================
    def ___online_backup____________________________________________():
        pass
        
    def _on_save_session_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle save session button click"""
        if not self.acp_client.session_id:
            msg_status(_('No active session to save'))
            return
        
        # Ask user for session name
        session_name = dlg_input(_('Session name:'), _('my-session'))
        if not session_name:
            return
        
        # Save session metadata
        session_data = {
            'session_id': self.acp_client.session_id,
            'agent_id': self.acp_client.agent_config['id'],
            'agent_name': self.acp_client.agent_config['name'],
            'name': session_name,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'cwd': os.getcwd()
        }
        
        # Get current working directory
        if ed.get_filename():
            session_data['cwd'] = os.path.dirname(ed.get_filename())
        
        # Save to config
        global _command_instance
        if _command_instance:
            _command_instance.save_session(session_data)
            msg_status(_(f'Session saved: {session_name}'))
            self._add_message(_(f'✓ Session saved as: {session_name}'), 'system')

    def _on_load_session_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle load session button click - auto-connect if needed"""
        global _command_instance
        if not _command_instance:
            return
        
        # Get list of saved sessions for THIS agent only
        current_agent_id = self.acp_client.agent_config['id']
        sessions = _command_instance.get_saved_sessions(current_agent_id)
        
        if not sessions:
            msg_status(_('No saved sessions found for this agent'))
            return
        
        # Build menu
        items = []
        for sess in sessions:
            timestamp = sess.get('timestamp', '')
            name = sess.get('name', sess.get('session_id', 'Unknown'))
            items.append(f"{name} ({timestamp})")
        
        # Show menu
        selected_idx = dlg_menu(DMENU_LIST, '\n'.join(items), caption=_('Load Session'))
        
        if selected_idx is not None and 0 <= selected_idx < len(sessions):
            selected_session = sessions[selected_idx]
            
            # Verify session is for current agent
            if selected_session.get('agent_id') != current_agent_id:
                msg_status(_('Session is not compatible with current agent'))
                return
            
            # Load session in background thread
            def load_thread():
                '''RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.'''
                try:
                    # Connect if not connected (AUTO-CONNECT)
                    if self.acp_client.state != 'connected':
                        self.message_queue.put(('system', 'Connecting to agent...'))
                        self.acp_client.establish_agent_connection()
                    
                    # Check if agent supports loading sessions
                    if not self.acp_client.supports_load_session:
                        self.message_queue.put(('agent_answer_error', 
                            'Agent does not support loading sessions'))
                        return
                    
                    # Load session
                    session_id = selected_session['session_id']
                    cwd = selected_session.get('cwd', os.getcwd())
                    
                    self.message_queue.put(('system', f'Loading session: {selected_session["name"]}'))
                    self.acp_client.load_session(session_id, cwd)
                    self.has_session = True
                    self.message_queue.put(('session_metadata', None))
                    self.message_queue.put(('system', '✓ Session loaded successfully'))
                    
                except Exception as e:
                    debug_print(f"ERROR: in load_thread: {e}")
                    self.message_queue.put(('agent_answer_error', f'Failed to load session: {str(e)}'))
                    if DEBUG:
                        import traceback
                        traceback.print_exc()
            
            threading.Thread(target=load_thread, daemon=True).start()
            
            # Ensure timer is running
            if _command_instance:
                _command_instance._ensure_timer_running()
                
    def _update_load_button_visibility(self):
        """Update Load button visibility based on saved sessions for current agent"""
        global _command_instance
        if not _command_instance:
            return
        
        # Check if there are saved sessions for this agent
        agent_id = self.acp_client.agent_config['id']
        sessions = _command_instance.get_saved_sessions(agent_id)
        
        # Show Load button if there are saved sessions
        has_saved_sessions = len(sessions) > 0
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_load_session, 
                prop={'vis': has_saved_sessions})
                
    # =============================================================================
    # offline session backup
    # =============================================================================
    def ___offline_backup____________________________________________():
        pass
        
    def _sanitize_filename(self, text: str, max_length: int = 50) -> str:
        """Sanitize text for use in filename"""
        # Remove/replace invalid characters
        invalid_chars = '<>:"/\\|?*\n\r\t'
        for char in invalid_chars:
            text = text.replace(char, '_')
        
        # Remove multiple spaces and trim
        text = ' '.join(text.split())
        
        # Limit length
        if len(text) > max_length:
            text = text[:max_length].rsplit(' ', 1)[0]  # Break at word boundary
        
        return text.strip('_. ')

    def _create_backup_file(self, first_question: str) -> str:
        """Create backup file and return its path"""
        global _command_instance
        if not _command_instance:
            return None
        
        # Create backup directory if needed
        backup_dir = os.path.join(app_path(APP_DIR_SETTINGS), 'ai_agents_backups')
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
        
        # Generate filename: YYYYMMDD_HHMMSS_agentname_question.md
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        agent_name = self._sanitize_filename(self.acp_client.agent_config['name'], 20)
        question_part = self._sanitize_filename(first_question, 30)
        
        filename = f"{timestamp}_{agent_name}_{question_part}.md"
        filepath = os.path.join(backup_dir, filename)
        
        debug_print(f"Created backup file: {filepath}")
        return filepath

    def _save_backup(self):
        """Save current chat to backup file"""
        if not self.backup_file:
            return
        
        try:
            # Get all text from messages editor
            content = self.ed_msg.get_text_all()
            
            # Add metadata header if file is new
            if not os.path.exists(self.backup_file) or os.path.getsize(self.backup_file) == 0:
                agent_name = self.acp_client.agent_config['name']
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                header = f"# AI Agent Chat Backup\n"
                header += f"**Agent:** {agent_name}\n"
                header += f"**Date:** {timestamp}\n"
                header += f"**Session ID:** {self.acp_client.session_id or 'N/A'}\n\n"
                header += "---\n\n"
                content = header + content
            
            # Write to file
            with open(self.backup_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            debug_print(f"Saved backup to: {self.backup_file}")
            
        except Exception as e:
            debug_print(f"Failed to save backup: {e}")

    def _finalize_backup(self):
        """Finalize backup when closing panel"""
        if self.backup_file:
            self._save_backup()
            debug_print(f"Finalized backup: {self.backup_file}")
            self.backup_file = None
            self.first_user_question = None

    def _on_backups_menu_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle backups menu button click - show dropdown menu"""
        global _command_instance
        if not _command_instance:
            return
        
        # Create popup menu
        h_menu = menu_proc(0, MENU_CREATE)
        
        # Add menu items
        menu_proc(h_menu, MENU_ADD, 
                 caption=_('List Chat Backups'), 
                 command=lambda: _command_instance.plugin_list_backups())
        
        menu_proc(h_menu, MENU_ADD, 
                 caption=_('Restore Chat Backup'), 
                 command=lambda: _command_instance.plugin_restore_backup())
        
        menu_proc(h_menu, MENU_ADD, 
                 caption=_('Open Backup Folder'), 
                 command=lambda: _command_instance.plugin_open_backup_folder())
        
        menu_proc(h_menu, MENU_ADD, caption='-')  # Separator
        
        # Add current chat option
        if self.backup_file:
            menu_proc(h_menu, MENU_ADD, 
                     caption=_('Open Current Chat Backup'), 
                     command=lambda: self._open_current_backup())
        
        # Show the menu at button position
        menu_proc(h_menu, MENU_SHOW)

    def _open_current_backup(self):
        """Open the current chat's backup file in editor"""
        if self.backup_file and os.path.exists(self.backup_file):
            file_open(self.backup_file)
            msg_status(_(f'Opened current backup'))
        else:
            msg_status(_('No backup file for current chat'))
             
    # =============================================================================
    # slash commands
    # =============================================================================
    def ___slash_command____________________________________________():
        pass
        
    def _on_commands_menu_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle commands menu button click - show dropdown menu of slash commands"""
        if not self.available_commands:
            msg_status(_('No commands available'))
            return
        
        # Create popup menu
        h_menu = menu_proc(0, MENU_CREATE)
        
        # Add menu items for each command
        for cmd in self.available_commands:
            name = cmd.get('name', '')
            description = cmd.get('description', '')
            input_spec = cmd.get('input')  # Can be None/null
            
            # Get input hint only if input_spec exists bcause hint may be null
            input_hint = ''
            if input_spec and isinstance(input_spec, dict):
                input_hint = input_spec.get('hint', '')
            
            # Build display text
            display = f"/{name}"
            if description:
                display += f" - {description}"
            
            # Create command handler
            def make_command_handler(cmd_name, cmd_input_hint):
                def handler():
                    self._execute_slash_command(cmd_name, cmd_input_hint)
                return handler
            
            menu_proc(h_menu, MENU_ADD, 
                     caption=display, 
                     command=make_command_handler(name, input_hint))
        
        # Show the menu
        menu_proc(h_menu, MENU_SHOW)

    def _execute_slash_command(self, command_name: str, input_hint: str = ''):
        """Execute a slash command - optionally prompt for input"""
        debug_print(f"Executing slash command: /{command_name}")
        
        # Check if command needs input
        command_text = f"/{command_name}"
        
        if input_hint:
            # Prompt user for input
            user_input = dlg_input(f'{input_hint}:', '')
            if user_input is None:  # User cancelled
                return
            if user_input:  # User provided input
                command_text += f" {user_input}"
        
        # Insert command into input box or send directly
        # Option 1: Insert into input box for user to review/edit
        current_text = self.ed_input.get_text_all()
        if current_text.strip():
            # Append to existing text
            self.ed_input.set_text_all(current_text + '\n' + command_text)
        else:
            # Set as new text
            self.ed_input.set_text_all(command_text)
        
        self.ed_input.focus()
        
        # Option 2: Send immediately (uncomment if you prefer auto-send)
        # self._handle_and_send_user_message(command_text)
        
    # =============================================================================
    # agent/model/mode/predefined model button menus
    # =============================================================================
    def __________agent_model_mode____________________________________________():
        pass
        
    def _on_agent_button_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle agent button click - show menu"""
        agents = get_agents_info()
        current_id = self.acp_client.agent_config['id']
        
        # Build menu
        h_menu = menu_proc(0, MENU_CREATE)
        
        for i, agent in enumerate(agents):
            name = agent['name']
            is_available = agent['available']
            is_current = (agent['id'] == current_id)
            
            # Format caption
            if is_current:
                caption = f'● {name}'
            else:
                caption = f'   {name}'
            
            if not is_available:
                caption += '    (not installed)'
            
            # Add menu item
            def make_handler(agent_data):
                def handler():
                    if agent_data['available']:
                        self.set_agent(agent_data)
                return handler
            
            menu_proc(h_menu, MENU_ADD, 
                     caption=caption,
                     command=make_handler(agent) if is_available else None)
        
        # Show menu
        menu_proc(h_menu, MENU_SHOW)

    def set_agent(self, agent):
        """Select an agent (called from menu)"""
        debug_print(f"User selected agent: {agent['name']}")
        
        # Save user's agent choice
        update_user_choice(agent['id'], 'last_agent_id', agent['id'])
        
        # Kill old agent if connected
        if self.acp_client.state != 'disconnected':
            self.acp_client.dispose()
        
        # Update agent config BEFORE reset
        self.acp_client.agent_config = agent
        
        # Update button caption
        truncated_name = self._truncate_text_for_button(agent["name"])
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_agent_button, 
                 prop={'cap': f'{truncated_name}'})
        
        # COMPREHENSIVE RESET
        self._reset_all_ui_and_state()
        
        # Apply saved settings for new agent
        self._apply_saved_settings()

    def _apply_saved_settings(self):
        """Apply saved user choices settings for this agent"""
        global _command_instance
        if not _command_instance:
            return
        
        agent_id = self.acp_client.agent_config.get('id')
        user_choices = _command_instance.user_choices
        agent_settings = user_choices.get('agent_settings', {}).get(agent_id, {})
        
        if not agent_settings:
            return
        
        debug_print(f"Applying saved settings for agent {agent_id}: {agent_settings}")
        
        # Store settings to apply after connection
        self._pending_settings = agent_settings.copy()
      
    def _on_predefined_model_button_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle predefined model button click - show menu"""
        agent = self.acp_client.agent_config
        
        if 'supported_models' not in agent or not agent['supported_models']:
            return
        
        models = agent['supported_models']
        
        # Get current model from saved preferences
        agent_id = agent.get('id')
        user_choices = load_user_choices()
        agent_settings = user_choices.get('agent_settings', {}).get(agent_id, {})
        current_model = agent_settings.get('predefined_model')
        
        # Build menu
        h_menu = menu_proc(0, MENU_CREATE)
        
        for model in models:
            model_id = model['id']
            model_name = model['name']
            is_current = (model_id == current_model)
            
            caption = f'● {model_name}' if is_current else f'   {model_name}'
            
            def make_handler(m_id, m_name):
                return lambda: self.set_predefined_model(m_id, m_name)
            
            menu_proc(h_menu, MENU_ADD, 
                     caption=caption,
                     command=make_handler(model_id, model_name))
        
        menu_proc(h_menu, MENU_SHOW)

    def _on_model_button_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle ACP model button click - show menu"""
        metadata = self.acp_client.session_metadata
        if not metadata:
            return
        
        models_data = metadata.get('models', {})
        items = models_data.get('availableModels', [])
        current_id = models_data.get('currentModelId')
        
        if not items:
            return
        
        # Build menu
        h_menu = menu_proc(0, MENU_CREATE)
        
        for model in items:
            model_id = model.get('modelId')
            model_name = model.get('name', model_id)
            description = model.get('description', '')
            is_current = (model_id == current_id)
            
            # Format caption with description
            caption = f'● {model_name}' if is_current else f'   {model_name}'
            if description:
                caption += f'\t      {description}'
            
            def make_handler(m_id, m_name):
                return lambda: self.set_model(m_id, m_name)
            
            menu_proc(h_menu, MENU_ADD,
                     caption=caption,
                     command=make_handler(model_id, model_name))
        
        menu_proc(h_menu, MENU_SHOW)

    def _on_mode_button_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle mode button click - show menu"""
        metadata = self.acp_client.session_metadata
        if not metadata:
            return
        
        modes_data = metadata.get('modes', {})
        items = modes_data.get('availableModes', [])
        current_id = modes_data.get('currentModeId')
        
        if not items:
            return
        
        # Build menu
        h_menu = menu_proc(0, MENU_CREATE)
        
        for mode in items:
            mode_id = mode.get('id')
            mode_name = mode.get('name', mode_id)
            description = mode.get('description', '')
            is_current = (mode_id == current_id)
            
            # Format caption with description
            caption = f'● {mode_name}' if is_current else f'   {mode_name}'
            if description:
                caption += f'\t      {description}'
            
            def make_handler(m_id, m_name):
                return lambda: self.set_mode(m_id, m_name)
            
            menu_proc(h_menu, MENU_ADD,
                     caption=caption,
                     command=make_handler(mode_id, mode_name))
        
        menu_proc(h_menu, MENU_SHOW)

    def _apply_pending_settings_after_connection(self):
        """
        Apply all saved user preferences after connection.
        """
        if not hasattr(self, '_pending_settings') or not self._pending_settings:
            return
        
        metadata = self.acp_client.session_metadata
        
        # Apply mode if exists
        if 'mode' in self._pending_settings and metadata and 'modes' in metadata:
            mode_id = self._pending_settings['mode']
            modes_data = metadata['modes']
            available_modes = modes_data.get('availableModes', [])
            
            # Find mode name
            for mode in available_modes:
                if mode.get('id') == mode_id:
                    self.set_mode(mode_id, mode.get('name', mode_id))
                    break
        
        # Apply model if exists
        if 'model' in self._pending_settings and metadata and 'models' in metadata:
            model_id = self._pending_settings['model']
            models_data = metadata['models']
            available_models = models_data.get('availableModels', [])
            
            # Find model name
            for model in available_models:
                if model.get('modelId') == model_id:
                    self.set_model(model_id, model.get('name', model_id))
                    break
        
        # NOTE: Predefined models are NOT applied here because they were already
        # applied before connection (in the agent command args)
        # Clear pending
        self._pending_settings = {}

    def _truncate_text_for_button(self, text, max_width_chars=17):
        """Truncate text to fit button width, add ellipsis"""
        if len(text) <= max_width_chars:
            return text
        return text[:max_width_chars-3] + "..."
    
    # === MODE MANAGEMENT =====================================================
    def ___mode______________________________________________________():
        pass
           
    def update_mode_ui(self, mode_id=None, mode_name=None):
        """
        Update mode button caption. Can be called with:
        - mode_id only (will look up name from metadata)
        - mode_id + mode_name (direct update, faster)
        - neither (will use currentModeId from metadata)
        
        This is the SINGLE function that updates the mode button.
        All other code paths call this.
        """
        metadata = self.acp_client.session_metadata
        if not metadata or 'modes' not in metadata:
            # No metadata yet, hide button
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_mode_button, 
                    prop={'vis': False})
            return
        
        modes_data = metadata['modes']
        available_modes = modes_data.get('availableModes', [])
        
        if not available_modes:
            # No modes available, hide button
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_mode_button, 
                    prop={'vis': False})
            return
        
        # Determine which mode to display
        if mode_id is None:
            mode_id = modes_data.get('currentModeId')
        
        # Look up name if not provided
        if mode_name is None and mode_id:
            for mode in available_modes:
                if mode.get('id') == mode_id:
                    mode_name = mode.get('name', mode_id)
                    break
        
        if mode_name is None:
            mode_name = 'Default'
        
        # Update button
        truncated_name = self._truncate_text_for_button(f'Mode: {mode_name}')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_mode_button, 
                prop={'cap': truncated_name, 'vis': True})
    
    def set_mode(self, mode_id, mode_name):
        """
        Change mode. This is the SINGLE function for ALL mode changes.
        Called from: 
        - _on_mode_button_click (user selection)
        - _apply_pending_settings_after_connection (restore saved preference)
        
        Does three things:
        1. Updates UI immediately (optimistic update)
        2. Saves user preference
        3. Sends to agent in background
        """
        debug_print(f"Changing mode to: {mode_id}")
        
        # 1. Update UI immediately (optimistic)
        self.update_mode_ui(mode_id, mode_name)
        
        # 2. Save user preference
        agent_id = self.acp_client.agent_config.get('id')
        update_user_choice(agent_id, 'mode', mode_id)
        
        # 3. Send to agent in background
        def send_mode_change():
            '''RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.'''
            try:
                self.acp_client.set_session_mode(mode_id)
                # Update local metadata to match
                if self.acp_client.session_metadata and 'modes' in self.acp_client.session_metadata:
                    modes = self.acp_client.session_metadata['modes']
                    if modes:
                        modes['currentModeId'] = mode_id
            except Exception as e:
                debug_print(f"Failed to send mode change: {e}")
                self.message_queue.put(('agent_answer_error', f'Failed to change mode: {str(e)}'))
        
        threading.Thread(target=send_mode_change, daemon=True).start()
    
    def _sync_mode_from_agent(self, mode_id):
        """
        Agent told us mode changed (via session/update notification).
        We DON'T send it back to agent (would create loop).
        Just update UI and local state.
        
        Called from: on_timer when receiving 'mode_changed' message
        """
        debug_print(f"Agent changed mode to: {mode_id}")
        
        # Update local metadata
        if self.acp_client.session_metadata and 'modes' in self.acp_client.session_metadata:
            modes = self.acp_client.session_metadata['modes']
            if modes:
                modes['currentModeId'] = mode_id
        
        # Update UI
        self.update_mode_ui(mode_id)
        
        # Optionally show message
        self._add_message(_(f'Mode changed to: {mode_id}'), 'system')

    # === MODEL MANAGEMENT (ACP) ==============================================
    def ___model______________________________________________________():
        pass

    def update_model_ui(self, model_id=None, model_name=None):
        """Update ACP model button caption. Single source of truth."""
        metadata = self.acp_client.session_metadata
        if not metadata or 'models' not in metadata:
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_model_button, 
                    prop={'vis': False})
            return
        
        models_data = metadata['models']
        available_models = models_data.get('availableModels', [])
        
        if not available_models:
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_model_button, 
                    prop={'vis': False})
            return
        
        if model_id is None:
            model_id = models_data.get('currentModelId')
        
        if model_name is None and model_id:
            for model in available_models:
                if model.get('modelId') == model_id:
                    model_name = model.get('name', model_id)
                    break
        
        if model_name is None:
            model_name = 'Default'
        
        truncated_name = self._truncate_text_for_button(f'Model: {model_name}')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_model_button, 
                prop={'cap': truncated_name, 'vis': True})
    
    def set_model(self, model_id, model_name):
        """
        Change model. This is the SINGLE function for ALL model changes.
        Called from:
        - _on_model_button_click (user selection)
        - _apply_pending_settings_after_connection (restore saved preference)
        """
        debug_print(f"Changing model to: {model_id}")
        
        # Update UI immediately
        self.update_model_ui(model_id, model_name)
        
        # Save preference
        agent_id = self.acp_client.agent_config.get('id')
        update_user_choice(agent_id, 'model', model_id)
        
        # Send to agent
        def send_model_change():
            '''RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.'''
            try:
                self.acp_client.set_session_model(model_id)
                if self.acp_client.session_metadata and 'models' in self.acp_client.session_metadata:
                    models = self.acp_client.session_metadata['models']
                    if models:
                        models['currentModelId'] = model_id
            except Exception as e:
                debug_print(f"Failed to send model change: {e}")
                self.message_queue.put(('agent_answer_error', f'Failed to change model: {str(e)}'))
        
        threading.Thread(target=send_model_change, daemon=True).start()

    # === PREDEFINED MODEL MANAGEMENT =========================================
    def ___pre_model______________________________________________________():
        pass

    def update_predefined_model_ui(self, model_id=None):
        """Update predefined model button caption. Single source of truth."""
        agent = self.acp_client.agent_config
        
        if 'supported_models' not in agent or not agent['supported_models']:
            dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_predefined_model_button, 
                    prop={'vis': False})
            return
        
        models = agent['supported_models']
        
        # Get current model from saved user preferences if not provided
        if model_id is None:
            agent_id = agent.get('id')
            user_choices = load_user_choices()
            agent_settings = user_choices.get('agent_settings', {}).get(agent_id, {})
            model_id = agent_settings.get('predefined_model')
        
        # Find name
        model_name = 'Default'
        if model_id:
            for model in models:
                if model['id'] == model_id:
                    model_name = model['name']
                    break
        
        truncated_name = self._truncate_text_for_button(f'Model: {model_name}')
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_predefined_model_button, 
                prop={'cap': truncated_name, 'vis': True})
    
    def set_predefined_model(self, model_id, model_name):
        """
        Change predefined model. This is the SINGLE function for ALL predefined model changes.
        Called from:
        - _on_predefined_model_button_click (user selection)
        - _apply_pending_settings_after_connection (restore saved preference)
        
        Note: For predefined models, we only save the preference and update UI.
        The actual model will be applied when reconnecting (via agent args or other mechanism).
        """
        debug_print(f"Changing predefined model to: {model_id}")
        
        # Update UI
        self.update_predefined_model_ui(model_id)
        
        # Save preference
        agent_id = self.acp_client.agent_config.get('id')
        update_user_choice(agent_id, 'predefined_model', model_id)
        
        # Warn if connected - model change requires reconnect
        if self.acp_client.state == 'connected':
            msg_status(_('Model changed. Please reconnect to apply.'))
            self._add_message(_('Model changed to: {}. Reconnect to apply this change.').format(model_name), 'system')

    def _apply_predefined_model_to_args(self):
        """
        Apply saved predefined model to agent args before connection.
        This modifies the command that will be executed.
        """
        agent = self.acp_client.agent_config
        
        # Only for agents with supported_models
        if 'supported_models' not in agent or not agent['supported_models']:
            return
        
        # Get saved preference
        agent_id = agent.get('id')
        user_choices = load_user_choices()
        agent_settings = user_choices.get('agent_settings', {}).get(agent_id, {})
        model_id = agent_settings.get('predefined_model')
        
        if not model_id:
            # No saved preference, don't add --model
            return
        
        # Verify model exists in supported_models
        model_exists = any(m['id'] == model_id for m in agent['supported_models'])
        if not model_exists:
            debug_print(f"Saved model {model_id} not found in supported models")
            return
        
        # Add or update --model in args
        args = agent.get('args', []).copy()
        
        if '--model' in args:
            # Update existing --model
            model_idx = args.index('--model')
            if model_idx + 1 < len(args):
                args[model_idx + 1] = model_id
            else:
                args.append(model_id)
        else:
            # Add new --model
            args.extend(['--model', model_id])
        
        agent['args'] = args
        debug_print(f"Applied predefined model to args: {model_id}")
        
    # =============================================================================
    def _________________________________________________________():
        pass
        
    def _reset_all_ui_and_state(self):
        """
        Comprehensive reset of ALL UI elements and state.
        Called when: switching agents, disconnect, kill, dispose, panel close
        """
        debug_print(f"Comprehensive reset for panel #{self.instance_id}")
        
        # ============================================================
        # RESET SESSION STATE
        # ============================================================
        self.has_session = False
        self.tools = {}
        self.agent_already_sent_first_msg = False
        self.first_user_question = None
        self.available_commands = []  # Reset commands
        
        # Reset ACP client session data
        if self.acp_client:
            self.acp_client.session_id = None
            self.acp_client.session_metadata = None
            self.acp_client.authenticated = False
            self.acp_client.supports_load_session = False
        
        # ============================================================
        # RESET COMMANDS BUTTON
        # ============================================================
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_commands_menu, prop={'vis': False})

        # ============================================================
        # RESET ALL button menu BOXES
        # ============================================================
        
        # Reset predefined model button menu
        self.update_predefined_model_ui()
        
        # Reset ACP model button menu
        self.update_model_ui()
        
        # Reset mode button menu
        self.update_mode_ui()
        
        # ============================================================
        # RESET SESSION BUTTONS
        # ============================================================
        
        # Hide Save button (no active session)
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_save_session, prop={'vis': False})
        
        # Update Load button based on current agent's saved sessions
        # This will hide it if switching to agent without saved sessions
        self._update_load_button_visibility()
        
        # ============================================================
        # RESET OPERATION BUTTONS
        # ============================================================
        
        # Hide Cancel button (no operation running)
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_cancel, prop={'vis': False})
        
        # Hide Kill button (no process running, will be updated by connection state)
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_kill, prop={'vis': False})
        
        # ============================================================
        # RESET CONNECT/DISCONNECT BUTTONS
        # ============================================================
        
        # Show Connect, hide Disconnect (disconnected state)
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_connect, prop={'vis': True})
        dlg_proc(self.h_dlg, DLG_CTL_PROP_SET, index=self.h_disconnect, prop={'vis': False})
        
        # ============================================================
        # UPDATE STATUSBAR
        # ============================================================
        self._update_statusbar()
        
        debug_print(f"Comprehensive reset completed for panel #{self.instance_id}")
       

# =============================================================================
# MAIN PLUGIN COMMAND
# =============================================================================

CELL_TAG_INFO = 20
CELL_TAG = app_proc(PROC_GET_UNIQUE_TAG, '')
BAR_H = app_proc(PROC_GET_MAIN_STATUSBAR, '')

fn_config = os.path.join(app_path(APP_DIR_SETTINGS), 'plugins.ini')
SECTION = 'ai_agents'

# Global instance for callbacks
_command_instance = None

class Command:
    """Main plugin command class - manages multiple chat panels"""
    
    title = "AI Agents"
    
    def __init__(self):
        debug_print("=== INITIALIZING AI Agents PLUGIN ===")
        
        self.chat_panels = []  # Track all open chat panels
        self.icon_index = -1
        self.timer_running = False
        
        # Store instance globally for timer callback
        global _command_instance
        _command_instance = self
        
        # Load options config
        self.load_ops()
        
        self.sessions_file = os.path.join(app_path(APP_DIR_SETTINGS), 'ai_agents_sessions.json')
        
        self.backup_dir = os.path.join(app_path(APP_DIR_SETTINGS), 'ai_agents_backups')

        # Create status bar item
        self.init_bar_cell()
        debug_print("Plugin initialized successfully")
        
    def _chat_timer(self, tag='', info=''):
        """Timer callback - checks all active panels"""
        for panel in self.chat_panels:
            try:
                panel.on_timer()
            except Exception as e:
                debug_print(f"Error in panel #{panel.instance_id} timer: {e}")
    
    def _ensure_timer_running(self):
        """Start the shared timer if not already running"""
        if not self.timer_running and self.chat_panels:
            debug_print("Starting shared timer")
            timer_proc(TIMER_START, TIMER_TAG, TIMER_INTERVAL)
            self.timer_running = True
    
    def _stop_timer(self):
        """Stop the shared timer"""
        if self.timer_running:
            debug_print("Stopping shared timer")
            timer_proc(TIMER_STOP, TIMER_TAG, interval=0)
            self.timer_running = False
        
    def load_ops(self):
        """Load plugin options from config"""
        # Load user choices
        self.user_choices = load_user_choices()
        debug_print(f"Loaded user choices: {self.user_choices}")

    def save_ops(self):
        """Save plugin options to config"""
        # User choices are now saved immediately when changed
        # TODO: remove this
        pass
        
    def save_session(self, session_data: Dict[str, Any]):
        """Save session metadata to disk"""
        # Load existing sessions
        sessions = self._load_sessions_file()
        
        # Add new session
        sessions.append(session_data)
        
        # Save to file
        try:
            with open(self.sessions_file, 'w', encoding='utf-8') as f:
                json.dump(sessions, f, indent=2)
            debug_print(f"Saved session: {session_data['session_id']}")
        except Exception as e:
            debug_print(f"Failed to save session: {e}")
    
    def get_saved_sessions(self, agent_id: str = None) -> List[Dict[str, Any]]:
        """Get list of saved sessions, optionally filtered by agent_id"""
        sessions = self._load_sessions_file()
        
        if agent_id:
            sessions = [s for s in sessions if s.get('agent_id') == agent_id]
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda s: s.get('timestamp', ''), reverse=True)
        
        return sessions
    
    def _load_sessions_file(self) -> List[Dict[str, Any]]:
        """Load sessions from disk"""
        if not os.path.exists(self.sessions_file):
            return []
        
        try:
            with open(self.sessions_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            debug_print(f"Failed to load sessions file: {e}")
            return []

    def init_bar_cell(self):
        """Initialize status bar cell for the plugin
        credits to cuda_git_status
        """
        # insert our cell before "info" cell
        
        # Find "info" cell index
        index_info = statusbar_proc(BAR_H, STATUSBAR_FIND_CELL, value=CELL_TAG_INFO)
        if index_info is None:
            return False
            
        # Check if our cell already exists
        index_new = statusbar_proc(BAR_H, STATUSBAR_FIND_CELL, value=CELL_TAG)
        if index_new is None:
            # Get colors from info cell
            old_color_back = statusbar_proc(BAR_H, STATUSBAR_GET_CELL_COLOR_BACK, tag=CELL_TAG_INFO)
            old_color_font = statusbar_proc(BAR_H, STATUSBAR_GET_CELL_COLOR_FONT, tag=CELL_TAG_INFO)
            
            # Add our cell before "info" cell
            statusbar_proc(BAR_H, STATUSBAR_ADD_CELL, index=index_info, tag=CELL_TAG)
            statusbar_proc(BAR_H, STATUSBAR_SET_CELL_COLOR_BACK, tag=CELL_TAG, value=old_color_back)
            statusbar_proc(BAR_H, STATUSBAR_SET_CELL_COLOR_FONT, tag=CELL_TAG, value=old_color_font)
            statusbar_proc(BAR_H, STATUSBAR_SET_CELL_ALIGN, tag=CELL_TAG, value='C')
            statusbar_proc(BAR_H, STATUSBAR_SET_CELL_AUTOSIZE, tag=CELL_TAG, value=True)
            statusbar_proc(BAR_H, STATUSBAR_SET_CELL_CALLBACK, tag=CELL_TAG, value='module=cuda_ai_agents;cmd=callback_statusbar_click;')
            
        # Load icon
        self._load_icon()
        
        # Set initial status
        self._update_main_statusbar()
        
        return True
        
    def _load_icon(self):
        """Load status bar icon"""
        imglist = statusbar_proc(BAR_H, STATUSBAR_GET_IMAGELIST)
        if not imglist:
            imglist = imagelist_proc(0, IMAGELIST_CREATE)
            statusbar_proc(BAR_H, STATUSBAR_SET_IMAGELIST, value=imglist)
            
        icon_name = 'ai-agents.png'
        # icon_name = 'qwen.png'
        # icon_name = 'gemini.png'
        fn_icon = os.path.join(os.path.dirname(__file__), icon_name)
        
        # If icon file doesn't exist, skip it
        if os.path.isfile(fn_icon):
            self.icon_index = imagelist_proc(imglist, IMAGELIST_ADD, value=fn_icon)
        else:
            self.icon_index = -1
            
    def _update_main_statusbar(self):
        """Update main status bar"""
        # Count connected panels
        connected_count = sum(1 for p in self.chat_panels if p.acp_client.state == 'connected')
        total_count = len(self.chat_panels)
        
        # ■⚪⚫⬤⭕⭘◯⚠💤✓✔✘✗•◦●○◍☀◬⚠
        if total_count == 0:
            text = 'AI'
            color = 0x808080
        elif connected_count == 0:
            text = f'AI ({total_count})'
            color = 0x808080
        elif connected_count < total_count:
            text = f'AI ({connected_count}/{total_count})'
            color = 0x0080FF
        else:
            text = f'AI ({total_count})'
            color = 0x00C000
        
        # Set text
        statusbar_proc(BAR_H, STATUSBAR_SET_CELL_TEXT, tag=CELL_TAG, value=text)
        
        # Set color
        # statusbar_proc(BAR_H, STATUSBAR_SET_CELL_COLOR_FONT, tag=CELL_TAG, value=color)
        
        # Set icon if available
        if self.icon_index >= 0:
            statusbar_proc(BAR_H, STATUSBAR_SET_CELL_IMAGEINDEX, tag=CELL_TAG, value=self.icon_index)
            
        # Auto-size the cell
        statusbar_proc(BAR_H, STATUSBAR_SET_CELL_AUTOSIZE, tag=CELL_TAG, value=True)
            
    def callback_statusbar_click(self, id_dlg, id_ctl, data='', info=''):
        """Handle main status bar click - open new chat window"""
        self.plugin_start_new_chat_window()

    def callback_panel_statusbar_click(self, *args, **kwargs):
        """Handle individual panel statusbar click - toggle that panel's window"""
        try:
            info = args[4] 
            instance_id = int(info)
            
            # Find the panel with this instance_id and toggle hide/show that specific panel
            for panel in self.chat_panels:
                if panel.instance_id == instance_id:
                    panel.toggle()
                    return
        except (ValueError, AttributeError, IndexError) as e:
            debug_print(f"Failed to handle statusbar click: {e}")
            debug_print(f"Failed to parse instance_id from info: {info}")
            
    def plugin_start_new_chat_window(self):
        """Open a new chat panel"""
        debug_print("Opening new chat panel")
        
        # Get initial agent config from user's last choice
        initial_agent = None
        last_agent_id = self.user_choices.get('last_agent_id')
        
        if last_agent_id:
            agents = get_agents_info()
            for agent in agents:
                if agent['id'] == last_agent_id and agent['available']:
                    initial_agent = agent
                    break
        
        if initial_agent is None:
            initial_agent = get_agents_info(return_first_available=True)
        
        # Create new panel
        panel = ChatPanel(initial_agent)
        self.chat_panels.append(panel)
        
        # Apply saved YOLO setting
        yolo_enabled = self.user_choices.get('yolo_enabled', False)
        dlg_proc(panel.h_dlg, DLG_CTL_PROP_SET, index=panel.h_yolo_check, 
                 prop={'val': 1 if yolo_enabled else 0})
        
        # Apply saved agent-specific settings (will be applied after connection)
        panel._apply_saved_settings()
        
        panel.show()
        
        # Ensure timer is running
        self._ensure_timer_running()
        
        self._update_main_statusbar()
            
    def plugin_list_all_agents(self):
        """Show a list of all running agents and allow selection"""
        debug_print("Command: list_all_agents")
        
        if not self.chat_panels:
            msg_status(_('No active agent windows'))
            return
        
        # Build list of agents with status icons
        items = []
        for panel in self.chat_panels:
            agent_name = panel.acp_client.agent_config['name']
            state = panel.acp_client.state
            
            # Status icons
            icon = status_icons.get(state, '◯')
            
            # Build display text
            text = f"{icon} {agent_name} #{panel.instance_id}"
            items.append(text)
        
        # Show menu and get selection
        selected_idx = dlg_menu(DMENU_LIST, '\n'.join(items), caption=_('Active AI Agents'))
        
        if selected_idx is not None and 0 <= selected_idx < len(self.chat_panels):
            # Show the selected panel
            panel = self.chat_panels[selected_idx]
            panel.show()
            
    def plugin_close_all_agents(self):
        """Close all agent windows and clean up everything"""
        debug_print("Command: close_all_agents")
        
        if not self.chat_panels:
            msg_status(_('No active agent windows'))
            return
        
        count = len(self.chat_panels)
        
        # Stop the shared timer first
        self._stop_timer()
        
        # Full cleanup for each panel
        # Kill all agents and clean up
        for panel in self.chat_panels[:]:  # Use slice to avoid modification during iteration
            try:
                # Kill the agent process
                if panel.acp_client:
                    panel.acp_client.kill_agent()
                
                # Remove statusbar
                panel._remove_statusbar()
                
                # Destroy dialog
                if panel.h_dlg:
                    dlg_proc(panel.h_dlg, DLG_FREE)
                    panel.h_dlg = None
            except Exception as e:
                debug_print(f"Error cleaning up panel #{panel.instance_id}: {e}")
        
        # Clear the list
        self.chat_panels.clear()
        
        # Update main statusbar
        self._update_main_statusbar()
        
        msg_status(_(f'Closed {count} agent window(s) and cleaned up'))
            
    def config(self):
        """Open plugin configuration"""
        self.save_ops()
        file_open(fn_config)
        
        # Try to jump to our section
        lines = [ed.get_text_line(i) for i in range(ed.get_line_count())]
        try:
            index = lines.index('[' + SECTION + ']')
            ed.set_caret(0, index)
        except:
            pass
            
    def plugin_open_config_file(self):
        """Open the plugin configuration file"""
        debug_print("Command: open_config_file")
        
        # Ensure config file exists
        load_and_merge_config()
        
        # Open in editor
        file_open(CONFIG_FILE)
        msg_status(_('Opened AI Agents config file'))
        
    def on_exit(self, ed_self):
        """Cleanup on plugin exit"""
        # Kill all agents
        for panel in self.chat_panels:
            try:
                if panel.acp_client:
                    panel.acp_client.kill_agent(0)
            except Exception as e:
                debug_print(f"Error killing agent on exit: {e}")
        
        debug_print("Plugin exiting, cleaning up...")
        self.save_ops()
        
        # Stop the shared timer
        self._stop_timer()
        
        self.chat_panels.clear()

    # =============================================================================
    # offline session backup
    # =============================================================================

    def plugin_list_backups(self):
        """List all chat backups and allow opening them"""
        debug_print("Command: list_backups")
        
        if not os.path.exists(self.backup_dir):
            msg_status(_('No backups found'))
            return
        
        # Get all .md files in backup directory
        backup_files = []
        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(self.backup_dir, filename)
                backup_files.append({
                    'name': filename,
                    'path': filepath,
                    'mtime': os.path.getmtime(filepath)
                })
        
        if not backup_files:
            msg_status(_('No backups found'))
            return
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x['mtime'], reverse=True)
        
        # Build menu
        items = []
        for backup in backup_files:
            # Parse filename: YYYYMMDD_HHMMSS_agentname_question.md
            name = backup['name'][:-3]  # Remove .md
            parts = name.split('_', 3)
            
            if len(parts) >= 3:
                date_part = parts[0]  # YYYYMMDD
                time_part = parts[1]  # HHMMSS
                agent_part = parts[2] if len(parts) > 2 else 'Unknown'
                question_part = parts[3] if len(parts) > 3 else ''
                
                # Format date/time
                try:
                    date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    time_str = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    display = f"[{date_str} {time_str}] {agent_part} - {question_part}"
                except:
                    display = name
            else:
                display = name
            
            items.append(display)
        
        # Show menu
        selected_idx = dlg_menu(DMENU_LIST, '\n'.join(items), 
                              caption=_('Chat Backups'))
        
        if selected_idx is not None and 0 <= selected_idx < len(backup_files):
            # Open selected backup in editor
            filepath = backup_files[selected_idx]['path']
            file_open(filepath)
            msg_status(_(f'Opened backup: {backup_files[selected_idx]["name"]}'))
    
    '''
    def plugin_open_backup_folder___have_access_violation_bug(self):
        """Open the backup folder in file manager"""
        debug_print("Command: open_backup_folder")
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        
        time.sleep(0.3)  # Small delay to trigguer the bug more frequently, without it the bug happen 1/10 times only, with this sleep it happens 8 or 9/10
        
        # Open folder in file manager
        if sys.platform == 'win32':
            os.startfile(self.backup_dir)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', self.backup_dir])
        else:
            subprocess.Popen(['xdg-open', self.backup_dir])
        
        msg_status(_(f'Opened backup folder'))
    '''
    
    def plugin_open_backup_folder(self):
        """Open the backup folder in file manager
        
        NB: os.startfile must be run inside a thread otherwise we get access violation errors related maybe to proc_timer because os.startfile seems to freeze the main thread for some milliseconds which prevent proc_timer from running correctly i think, see notes above in "fix Access violation error in cudatext" for more details
        """
        debug_print("Command: open_backup_folder")
        
        # Create directory if it doesn't exist
        if not os.path.exists(self.backup_dir):
            os.makedirs(self.backup_dir)
        
        def open_dir():
            '''RUNS IN WORKER THREAD! DO NOT CALL CUDATEXT API HERE! USE QUEUE INSTEAD.'''
            # Open folder in file manager
            if sys.platform == 'win32':
                os.startfile(self.backup_dir)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', self.backup_dir])
            else:
                subprocess.Popen(['xdg-open', self.backup_dir])
            
            msg_status(_(f'Opened backup folder'))

        # Launch in background thread - main thread returns immediately
        threading.Thread(target=open_dir, daemon=True).start()

    
    def plugin_restore_backup(self):
        """Restore a backup into a new chat window"""
        debug_print("Command: restore_backup")
        
        if not os.path.exists(self.backup_dir):
            msg_status(_('No backups found'))
            return
        
        # Get all .md files
        backup_files = []
        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(self.backup_dir, filename)
                backup_files.append({
                    'name': filename,
                    'path': filepath,
                    'mtime': os.path.getmtime(filepath)
                })
        
        if not backup_files:
            msg_status(_('No backups found'))
            return
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x['mtime'], reverse=True)
        
        # Build menu
        items = []
        for backup in backup_files:
            name = backup['name'][:-3]
            parts = name.split('_', 3)
            
            if len(parts) >= 3:
                date_part = parts[0]
                time_part = parts[1]
                agent_part = parts[2] if len(parts) > 2 else 'Unknown'
                question_part = parts[3] if len(parts) > 3 else ''
                
                try:
                    date_str = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                    time_str = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                    display = f"[{date_str} {time_str}] {agent_part} - {question_part}"
                except:
                    display = name
            else:
                display = name
            
            items.append(display)
        
        # Show menu
        selected_idx = dlg_menu(DMENU_LIST, '\n'.join(items), 
                              caption=_('Restore Chat Backup'))
        
        if selected_idx is not None and 0 <= selected_idx < len(backup_files):
            filepath = backup_files[selected_idx]['path']
            
            # Read backup content
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Determine agent from filename
                name = backup_files[selected_idx]['name'][:-3]
                parts = name.split('_', 3)
                agent_name = parts[2] if len(parts) > 2 else None
                
                # Find matching agent
                initial_agent = None
                if agent_name:
                    agents = get_agents_info()
                    for agent in agents:
                        if self._sanitize_agent_name(agent['name']) == agent_name and agent['available']:
                            initial_agent = agent
                            break
                
                if not initial_agent:
                    initial_agent = get_agents_info(return_first_available=True)
                
                # Create new panel with restored content
                panel = ChatPanel(initial_agent)
                self.chat_panels.append(panel)
                
                # Load content into messages editor
                panel.ed_msg.set_prop(PROP_RO, False)
                panel.ed_msg.set_text_all(content)
                panel.ed_msg.set_prop(PROP_RO, True)
                
                # Show panel
                panel.show()
                
                # Ensure timer is running
                self._ensure_timer_running()
                self._update_main_statusbar()
                
                msg_status(_(f'Restored backup: {backup_files[selected_idx]["name"]}'))
                
            except Exception as e:
                debug_print(f"Failed to restore backup: {e}")
                msg_status(_(f'Failed to restore backup: {e}'))
    
    def _sanitize_agent_name(self, name: str) -> str:
        """Sanitize agent name for comparison"""
        invalid_chars = '<>:"/\\|?*\n\r\t'
        for char in invalid_chars:
            name = name.replace(char, '_')
        return ' '.join(name.split()).strip('_. ')[:20]
