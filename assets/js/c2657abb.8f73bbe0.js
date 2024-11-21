"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[15572],{98962:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>o,default:()=>h,frontMatter:()=>s,metadata:()=>r,toc:()=>l});var i=t(85893),a=t(11151);const s={sidebar_label:"captainagent",title:"agentchat.contrib.captainagent"},o=void 0,r={id:"reference/agentchat/contrib/captainagent",title:"agentchat.contrib.captainagent",description:"CaptainAgent",source:"@site/docs/reference/agentchat/contrib/captainagent.md",sourceDirName:"reference/agentchat/contrib",slug:"/reference/agentchat/contrib/captainagent",permalink:"/ag2/docs/reference/agentchat/contrib/captainagent",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/website/docs/reference/agentchat/contrib/captainagent.md",tags:[],version:"current",frontMatter:{sidebar_label:"captainagent",title:"agentchat.contrib.captainagent"},sidebar:"referenceSideBar",previous:{title:"agent_optimizer",permalink:"/ag2/docs/reference/agentchat/contrib/agent_optimizer"},next:{title:"gpt_assistant_agent",permalink:"/ag2/docs/reference/agentchat/contrib/gpt_assistant_agent"}},c={},l=[{value:"CaptainAgent",id:"captainagent",level:2},{value:"__init__",id:"__init__",level:3},{value:"CaptainUserProxyAgent",id:"captainuserproxyagent",level:2},{value:"__init__",id:"__init__-1",level:3}];function d(e){const n={a:"a",code:"code",em:"em",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,a.a)(),...e.components};return(0,i.jsxs)(i.Fragment,{children:[(0,i.jsx)(n.h2,{id:"captainagent",children:"CaptainAgent"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class CaptainAgent(ConversableAgent)\n"})}),"\n",(0,i.jsx)(n.p,{children:"(In preview) Captain agent, designed to solve a task with an agent or a group of agents."}),"\n",(0,i.jsx)(n.h3,{id:"__init__",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'def __init__(name: str,\n             system_message: Optional[str] = None,\n             llm_config: Optional[Union[Dict, Literal[False]]] = None,\n             is_termination_msg: Optional[Callable[[Dict], bool]] = None,\n             max_consecutive_auto_reply: Optional[int] = None,\n             human_input_mode: Optional[str] = "NEVER",\n             code_execution_config: Optional[Union[Dict,\n                                                   Literal[False]]] = False,\n             nested_config: Optional[Dict] = None,\n             description: Optional[str] = DEFAULT_DESCRIPTION,\n             **kwargs)\n'})}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"name"})," ",(0,i.jsx)(n.em,{children:"str"})," - agent name."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"system_message"})," ",(0,i.jsx)(n.em,{children:"str"})," - system message for the ChatCompletion inference.\nPlease override this attribute if you want to reprogram the agent."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"llm_config"})," ",(0,i.jsx)(n.em,{children:"dict"})," - llm inference configuration.\nPlease refer to ",(0,i.jsx)(n.a,{href:"/docs/reference/oai/client#create",children:"OpenAIWrapper.create"})," for available options."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"is_termination_msg"})," ",(0,i.jsx)(n.em,{children:"function"}),' - a function that takes a message in the form of a dictionary\nand returns a boolean value indicating if this received message is a termination message.\nThe dict can contain the following keys: "content", "role", "name", "function_call".']}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"max_consecutive_auto_reply"})," ",(0,i.jsx)(n.em,{children:"int"}),' - the maximum number of consecutive auto replies.\ndefault to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).\nThe limit only plays a role when human_input_mode is not "ALWAYS".']}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"**kwargs"})," ",(0,i.jsx)(n.em,{children:"dict"})," - Please refer to other kwargs in\n",(0,i.jsx)(n.a,{href:"https://github.com/ag2ai/ag2/blob/main/autogen/agentchat/conversable_agent.py#L74",children:"ConversableAgent"}),"."]}),"\n"]}),"\n",(0,i.jsx)(n.h2,{id:"captainuserproxyagent",children:"CaptainUserProxyAgent"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:"class CaptainUserProxyAgent(ConversableAgent)\n"})}),"\n",(0,i.jsx)(n.p,{children:"(In preview) A proxy agent for the captain agent, that can execute code and provide feedback to the other agents."}),"\n",(0,i.jsx)(n.h3,{id:"__init__-1",children:"__init__"}),"\n",(0,i.jsx)(n.pre,{children:(0,i.jsx)(n.code,{className:"language-python",children:'def __init__(name: str,\n             nested_config: Dict,\n             agent_config_save_path: str = None,\n             is_termination_msg: Optional[Callable[[Dict], bool]] = None,\n             max_consecutive_auto_reply: Optional[int] = None,\n             human_input_mode: Optional[str] = "NEVER",\n             code_execution_config: Optional[Union[Dict,\n                                                   Literal[False]]] = None,\n             default_auto_reply: Optional[Union[str, Dict,\n                                                None]] = DEFAULT_AUTO_REPLY,\n             llm_config: Optional[Union[Dict, Literal[False]]] = False,\n             system_message: Optional[Union[str, List]] = "",\n             description: Optional[str] = None)\n'})}),"\n",(0,i.jsxs)(n.p,{children:[(0,i.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"name"})," ",(0,i.jsx)(n.em,{children:"str"})," - name of the agent."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"nested_config"})," ",(0,i.jsx)(n.em,{children:"dict"})," - the configuration for the nested chat instantiated by CaptainAgent."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"is_termination_msg"})," ",(0,i.jsx)(n.em,{children:"function"}),' - a function that takes a message in the form of a dictionary\nand returns a boolean value indicating if this received message is a termination message.\nThe dict can contain the following keys: "content", "role", "name", "function_call".']}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"max_consecutive_auto_reply"})," ",(0,i.jsx)(n.em,{children:"int"}),' - the maximum number of consecutive auto replies.\ndefault to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).\nThe limit only plays a role when human_input_mode is not "ALWAYS".']}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"human_input_mode"})," ",(0,i.jsx)(n.em,{children:"str"}),' - whether to ask for human inputs every time a message is received.\nPossible values are "ALWAYS", "TERMINATE", "NEVER".\n(1) When "ALWAYS", the agent prompts for human input every time a message is received.\nUnder this mode, the conversation stops when the human input is "exit",\nor when is_termination_msg is True and there is no human input.\n(2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or\nthe number of auto reply reaches the max_consecutive_auto_reply.\n(3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops\nwhen the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.']}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"code_execution_config"})," ",(0,i.jsx)(n.em,{children:"dict or False"})," - config for the code execution.\nTo disable code execution, set to False. Otherwise, set to a dictionary with the following keys:\n",(0,i.jsxs)(n.ul,{children:["\n",(0,i.jsx)(n.li,{children:'work_dir (Optional, str): The working directory for the code execution.\nIf None, a default working directory will be used.\nThe default working directory is the "extensions" directory under\n"path_to_autogen".'}),"\n",(0,i.jsx)(n.li,{children:"use_docker (Optional, list, str or bool): The docker image to use for code execution.\nDefault is True, which means the code will be executed in a docker container. A default list of images will be used.\nIf a list or a str of image name(s) is provided, the code will be executed in a docker container\nwith the first image successfully pulled.\nIf False, the code will be executed in the current environment.\nWe strongly recommend using docker for code execution."}),"\n",(0,i.jsx)(n.li,{children:"timeout (Optional, int): The maximum execution time in seconds."}),"\n",(0,i.jsx)(n.li,{children:"last_n_messages (Experimental, Optional, int): The number of messages to look back for code execution. Default to 1."}),"\n"]}),"\n"]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"default_auto_reply"})," ",(0,i.jsx)(n.em,{children:"str or dict or None"})," - the default auto reply message when no code execution or llm based reply is generated."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"llm_config"})," ",(0,i.jsx)(n.em,{children:"dict or False"})," - llm inference configuration.\nPlease refer to ",(0,i.jsx)(n.a,{href:"/docs/reference/oai/client#create",children:"OpenAIWrapper.create"}),"\nfor available options.\nDefault to false, which disables llm-based auto reply."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"system_message"})," ",(0,i.jsx)(n.em,{children:"str or List"})," - system message for ChatCompletion inference.\nOnly used when llm_config is not False. Use it to reprogram the agent."]}),"\n",(0,i.jsxs)(n.li,{children:[(0,i.jsx)(n.code,{children:"description"})," ",(0,i.jsx)(n.em,{children:"str"})," - a short description of the agent. This description is used by other agents\n(e.g. the GroupChatManager) to decide when to call upon this agent. (Default: system_message)"]}),"\n"]})]})}function h(e={}){const{wrapper:n}={...(0,a.a)(),...e.components};return n?(0,i.jsx)(n,{...e,children:(0,i.jsx)(d,{...e})}):d(e)}},11151:(e,n,t)=>{t.d(n,{Z:()=>r,a:()=>o});var i=t(67294);const a={},s=i.createContext(a);function o(e){const n=i.useContext(s);return i.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:o(e.components),i.createElement(s.Provider,{value:n},e.children)}}}]);