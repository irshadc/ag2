"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[80948],{37978:(n,e,t)=>{t.r(e),t.d(e,{assets:()=>c,contentTitle:()=>r,default:()=>g,frontMatter:()=>o,metadata:()=>i,toc:()=>l});var a=t(85893),s=t(11151);const o={custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/website/docs/topics/swarm.ipynb",source_notebook:"/website/docs/topics/swarm.ipynb",title:"Swarm Ochestration"},r="Swarm Ochestration",i={id:"topics/swarm",title:"Swarm Ochestration",description:"Open In Colab",source:"@site/docs/topics/swarm.mdx",sourceDirName:"topics",slug:"/topics/swarm",permalink:"/ag2/docs/topics/swarm",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/website/docs/topics/swarm.ipynb",tags:[],version:"current",frontMatter:{custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/website/docs/topics/swarm.ipynb",source_notebook:"/website/docs/topics/swarm.ipynb",title:"Swarm Ochestration"},sidebar:"docsSidebar",previous:{title:"Retrieval Augmentation",permalink:"/ag2/docs/topics/retrieval_augmentation"},next:{title:"Task Decomposition",permalink:"/ag2/docs/topics/task_decomposition"}},c={},l=[{value:"Components",id:"components",level:2},{value:"Create a <code>SwarmAgent</code>",id:"create-a-swarmagent",level:3},{value:"Registering Handoffs",id:"registering-handoffs",level:3},{value:"AFTER_WORK",id:"after_work",level:3},{value:"Initialize SwarmChat with <code>initiate_swarm_chat</code>",id:"initialize-swarmchat-with-initiate_swarm_chat",level:3},{value:"Q&amp;As",id:"qas",level:2},{value:"Demonstration",id:"demonstration",level:2},{value:"Create Swarm Agents",id:"create-swarm-agents",level:3},{value:"Start Chat",id:"start-chat",level:3},{value:"Demo with User Agent",id:"demo-with-user-agent",level:3}];function h(n){const e={a:"a",blockquote:"blockquote",code:"code",h1:"h1",h2:"h2",h3:"h3",img:"img",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,s.a)(),...n.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(e.h1,{id:"swarm-ochestration",children:"Swarm Ochestration"}),"\n",(0,a.jsxs)(e.p,{children:[(0,a.jsx)(e.a,{href:"https://colab.research.google.com/github/ag2ai/ag2/blob/main/website/docs/topics/swarm.ipynb",children:(0,a.jsx)(e.img,{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Open In Colab"})}),"\n",(0,a.jsx)(e.a,{href:"https://github.com/ag2ai/ag2/blob/main/website/docs/topics/swarm.ipynb",children:(0,a.jsx)(e.img,{src:"https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github",alt:"Open on GitHub"})})]}),"\n",(0,a.jsxs)(e.p,{children:["With AG2, you can initiate a Swarm Chat similar to OpenAI\u2019s\n",(0,a.jsx)(e.a,{href:"https://github.com/openai/swarm",children:"Swarm"}),". This orchestration offers two\nmain features:"]}),"\n",(0,a.jsxs)(e.ul,{children:["\n",(0,a.jsxs)(e.li,{children:[(0,a.jsx)(e.strong,{children:"Headoffs"}),": Agents can transfer control to another agent via\nfunction calls, enabling smooth transitions within workflows."]}),"\n",(0,a.jsxs)(e.li,{children:[(0,a.jsx)(e.strong,{children:"Context Variables"}),": Agents can dynamically update shared\nvariables through function calls, maintaining context and\nadaptability throughout the process."]}),"\n"]}),"\n",(0,a.jsx)(e.p,{children:"Instead of sending a task to a single LLM agent, you can assign it to a\nswarm of agents. Each agent in the swarm can decide whether to hand off\nthe task to another agent. The chat terminates when the last active\nagent\u2019s response is a plain string (i.e., it doesn\u2019t suggest a tool call\nor handoff)."}),"\n",(0,a.jsx)(e.h2,{id:"components",children:"Components"}),"\n",(0,a.jsx)(e.p,{children:"We now introduce the main components that need to be used to create a\nswarm chat."}),"\n",(0,a.jsxs)(e.h3,{id:"create-a-swarmagent",children:["Create a ",(0,a.jsx)(e.code,{children:"SwarmAgent"})]}),"\n",(0,a.jsxs)(e.p,{children:["All the agents passed to the swarm chat should be instances of\n",(0,a.jsx)(e.code,{children:"SwarmAgent"}),". ",(0,a.jsx)(e.code,{children:"SwarmAgent"})," is very similar to ",(0,a.jsx)(e.code,{children:"AssistantAgent"}),", but it\nhas some additional features to support function registration and\nhandoffs. When creating a ",(0,a.jsx)(e.code,{children:"SwarmAgent"}),", you can pass in a list of\nfunctions. These functions will be converted to schemas to be passed to\nthe LLMs, and you don\u2019t need to worry about registering the functions\nfor execution. You can also pass back a ",(0,a.jsx)(e.code,{children:"SwarmResult"})," class, where you\ncan return a value, the next agent to call, and update context variables\nat the same time."]}),"\n",(0,a.jsx)(e.p,{children:(0,a.jsx)(e.strong,{children:"Notes for creating the function calls"})}),"\n",(0,a.jsxs)(e.ul,{children:["\n",(0,a.jsxs)(e.li,{children:["For input arguments, you must define the type of the argument,\notherwise, the registration will fail (e.g.\xa0",(0,a.jsx)(e.code,{children:"arg_name: str"}),")."]}),"\n",(0,a.jsxs)(e.li,{children:["If your function requires access or modification of the context\nvariables, you must pass in ",(0,a.jsx)(e.code,{children:"context_variables: dict"})," as one argument.\nThis argument will not be visible to the LLM (removed when registering\nthe function schema). But when called, the global context variables will\nbe passed in by the swarm chat."]}),"\n",(0,a.jsx)(e.li,{children:"The docstring of the function will be used as the prompt. So make sure\nto write a clear description."}),"\n",(0,a.jsx)(e.li,{children:"The function name will be used as the tool name."}),"\n"]}),"\n",(0,a.jsx)(e.h3,{id:"registering-handoffs",children:"Registering Handoffs"}),"\n",(0,a.jsxs)(e.p,{children:["While you can create a function to decide what next agent to call, we\nprovide a quick way to register the handoff using ",(0,a.jsx)(e.code,{children:"ON_CONDITION"}),". We\nwill craft this transition function and add it to the LLM config\ndirectly."]}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'agent_2 = SwarmAgent(...)\nagent_3 = SwarmAgent(...)\n\n# Register the handoff\nagent_1 = SwarmAgent(...)\nagent_1.handoff(hand_to=[ON_CONDITION(agent_2, "condition_1"), ON_CONDITION(agent_3, "condition_2")])\n\n# This is equivalent to:\ndef transfer_to_agent_2():\n    """condition_1"""\n    return agent_2\n\ndef transfer_to_agent_3():\n    """condition_1"""\n    return agent_3\n    \nagent_1 = SwarmAgent(..., functions=[transfer_to_agent_2, transfer_to_agent_3])\n# You can also use agent_1.add_functions to add more functions after initialization\n'})}),"\n",(0,a.jsx)(e.h3,{id:"after_work",children:"AFTER_WORK"}),"\n",(0,a.jsxs)(e.p,{children:["When the last active agent\u2019s response doesn\u2019t suggest a tool call or\nhandoff, the chat will terminate by default. However, you can register\nan ",(0,a.jsx)(e.code,{children:"AFTER_WORK"})," handoff to define a fallback agent if you don\u2019t want the\nchat to end at this agent. At the swarm chat level, you also pass in an\n",(0,a.jsx)(e.code,{children:"AFTER_WORK"})," handoff to define the fallback mechanism for the entire\nchat. If this parameter is set for the agent and the chat, we will\nprioritize the agent\u2019s setting. There should only be one ",(0,a.jsx)(e.code,{children:"AFTER_WORK"}),".\nIf multiple ",(0,a.jsx)(e.code,{children:"AFTER_WORK"})," handoffs are passed, only the last one will be\nused."]}),"\n",(0,a.jsxs)(e.p,{children:["Besides fallback to an agent, we provide 3 ",(0,a.jsx)(e.code,{children:"AfterWorkOption"}),":"]}),"\n",(0,a.jsxs)(e.ul,{children:["\n",(0,a.jsxs)(e.li,{children:[(0,a.jsx)(e.code,{children:"TERMINATE"}),": Terminate the chat"]}),"\n",(0,a.jsxs)(e.li,{children:[(0,a.jsx)(e.code,{children:"STAY"}),": Stay at the current agent"]}),"\n",(0,a.jsxs)(e.li,{children:[(0,a.jsx)(e.code,{children:"REVERT_TO_USER"}),": Revert to the user agent. Only if a user agent is\npassed in when initializing. (See below for more details)"]}),"\n"]}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:"agent_1 = SwarmAgent(...)\n\n# Register the handoff\nagent_1.handoff(hand_to=[\n ON_CONDITION(...), \n ON_CONDITION(...),\n AFTER_WORK(agent_4) # Fallback to agent_4 if no handoff is suggested\n])\n\nagent_2.handoff(hand_to=[AFTER_WORK(AfterWorkOption.TERMINATE)]) # Terminate the chat if no handoff is suggested\n"})}),"\n",(0,a.jsxs)(e.h3,{id:"initialize-swarmchat-with-initiate_swarm_chat",children:["Initialize SwarmChat with ",(0,a.jsx)(e.code,{children:"initiate_swarm_chat"})]}),"\n",(0,a.jsxs)(e.p,{children:["After a set of swarm agents are created, you can initiate a swarm chat\nby calling ",(0,a.jsx)(e.code,{children:"initiate_swarm_chat"}),"."]}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'chat_history, context_variables, last_active_agent = initiate_swarm_chat(\n    init_agent=agent_1, # the first agent to start the chat\n    agents=[agent_1, agent_2, agent_3], # a list of agents\n    messages=[{"role": "user", "content": "Hello"}], # a list of messages to start the chat\n    user_agent=user_agent, # optional, if you want to revert to the user agent\n    context_variables={"key": "value"} # optional, initial context variables\n)\n'})}),"\n",(0,a.jsx)(e.h2,{id:"qas",children:"Q&As"}),"\n",(0,a.jsxs)(e.blockquote,{children:["\n",(0,a.jsx)(e.p,{children:"How are context variables updated?"}),"\n"]}),"\n",(0,a.jsxs)(e.p,{children:["The context variables will only be updated through custom function calls\nwhen returning a ",(0,a.jsx)(e.code,{children:"SwarmResult"})," object. In fact, all interactions with\ncontext variables will be done through function calls (accessing and\nupdating). The context variables dictionary is a reference, and any\nmodification will be done in place."]}),"\n",(0,a.jsxs)(e.blockquote,{children:["\n",(0,a.jsx)(e.p,{children:"What is the difference between ON_CONDITION and AFTER_WORK?"}),"\n"]}),"\n",(0,a.jsx)(e.p,{children:"When registering an ON_CONDITION handoff, we are creating a function\nschema to be passed to the LLM. The LLM will decide whether to call this\nfunction."}),"\n",(0,a.jsx)(e.p,{children:"When registering an AFTER_WORK handoff, we are defining the fallback\nmechanism when no tool calls are suggested. This is a higher level of\ncontrol from the swarm chat level."}),"\n",(0,a.jsxs)(e.blockquote,{children:["\n",(0,a.jsx)(e.p,{children:"When to pass in a user agent?"}),"\n"]}),"\n",(0,a.jsx)(e.p,{children:"If your application requires interactions with the user, you can pass in\na user agent to the groupchat, so that don\u2019t need to write an outer loop\nto accept user inputs and call swarm."}),"\n",(0,a.jsx)(e.h2,{id:"demonstration",children:"Demonstration"}),"\n",(0,a.jsx)(e.h3,{id:"create-swarm-agents",children:"Create Swarm Agents"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'import autogen\n\nconfig_list = autogen.config_list_from_json(...)\nllm_config = {"config_list": config_list}\n'})}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'import random\n\nfrom autogen import (\n    AFTER_WORK,\n    ON_CONDITION,\n    AfterWorkOption,\n    SwarmAgent,\n    SwarmResult,\n    initiate_swarm_chat,\n)\n\n\n# 1. A function that returns a value of "success" and updates the context variable "1" to True\ndef update_context_1(context_variables: dict) -> str:\n    context_variables["1"] = True\n    return SwarmResult(value="success", context_variables=context_variables)\n\n\n# 2. A function that returns an SwarmAgent object\ndef transfer_to_agent_2() -> SwarmAgent:\n    """Transfer to agent 2"""\n    return agent_2\n\n\n# 3. A function that returns the value of "success", updates the context variable and transfers to agent 3\ndef update_context_2_and_transfer_to_3(context_variables: dict) -> str:\n    context_variables["2"] = True\n    return SwarmResult(value="success", context_variables=context_variables, agent=agent_3)\n\n\n# 4. A function that returns a normal value\ndef get_random_number() -> str:\n    return random.randint(1, 100)\n\n\ndef update_context_3_with_random_number(context_variables: dict, random_number: int) -> str:\n    context_variables["3"] = random_number\n    return SwarmResult(value="success", context_variables=context_variables)\n\n\nagent_1 = SwarmAgent(\n    name="Agent_1",\n    system_message="You are Agent 1, first, call the function to update context 1, and transfer to Agent 2",\n    llm_config=llm_config,\n    functions=[update_context_1, transfer_to_agent_2],\n)\n\nagent_2 = SwarmAgent(\n    name="Agent_2",\n    system_message="You are Agent 2, call the function that updates context 2 and transfer to Agent 3",\n    llm_config=llm_config,\n    functions=[update_context_2_and_transfer_to_3],\n)\n\nagent_3 = SwarmAgent(\n    name="Agent_3",\n    system_message="You are Agent 3, tell a joke",\n    llm_config=llm_config,\n)\n\nagent_4 = SwarmAgent(\n    name="Agent_4",\n    system_message="You are Agent 4, call the function to get a random number",\n    llm_config=llm_config,\n    functions=[get_random_number],\n)\n\nagent_5 = SwarmAgent(\n    name="Agent_5",\n    system_message="Update context 3 with the random number.",\n    llm_config=llm_config,\n    functions=[update_context_3_with_random_number],\n)\n\n\n# This is equivalent to writing a transfer function\nagent_3.register_hand_off(ON_CONDITION(agent_4, "Transfer to Agent 4"))\n\nagent_4.register_hand_off([AFTER_WORK(agent_5)])\n\nprint("Agent 1 function schema:")\nfor func_schema in agent_1.llm_config["tools"]:\n    print(func_schema)\n\nprint("Agent 3 function schema:")\nfor func_schema in agent_3.llm_config["tools"]:\n    print(func_schema)\n'})}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-text",children:"Agent 1 function schema:\n{'type': 'function', 'function': {'description': '', 'name': 'update_context_1', 'parameters': {'type': 'object', 'properties': {}}}}\n{'type': 'function', 'function': {'description': 'Transfer to agent 2', 'name': 'transfer_to_agent_2', 'parameters': {'type': 'object', 'properties': {}}}}\nAgent 3 function schema:\n{'type': 'function', 'function': {'description': 'Transfer to Agent 4', 'name': 'transfer_to_Agent_4', 'parameters': {'type': 'object', 'properties': {}}}}\n"})}),"\n",(0,a.jsx)(e.h3,{id:"start-chat",children:"Start Chat"}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'context_variables = {"1": False, "2": False, "3": False}\nchat_result, context_variables, last_agent = initiate_swarm_chat(\n    init_agent=agent_1,\n    agents=[agent_1, agent_2, agent_3, agent_4, agent_5],\n    messages="start",\n    context_variables=context_variables,\n    after_work=AFTER_WORK(AfterWorkOption.TERMINATE),  # this is the default\n)\n'})}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-text",children:'Agent_1 (to chat_manager):\n\nstart\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_1\n\nAgent_1 (to chat_manager):\n\n***** Suggested tool call (call_kfcEAY2IeRZww06CQN7lbxOf): update_context_1 *****\nArguments: \n{}\n*********************************************************************************\n***** Suggested tool call (call_izl5eyV8IQ0Wg6XY2SaR1EJM): transfer_to_agent_2 *****\nArguments: \n{}\n************************************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Tool_Execution\n\n\n>>>>>>>> EXECUTING FUNCTION update_context_1...\n\n>>>>>>>> EXECUTING FUNCTION transfer_to_agent_2...\nTool_Execution (to chat_manager):\n\n***** Response from calling tool (call_kfcEAY2IeRZww06CQN7lbxOf) *****\n\n**********************************************************************\n\n--------------------------------------------------------------------------------\n***** Response from calling tool (call_izl5eyV8IQ0Wg6XY2SaR1EJM) *****\nSwarmAgent --\x3e Agent_2\n**********************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_2\n\nAgent_2 (to chat_manager):\n\n***** Suggested tool call (call_Yf5DTGaaYkA726ubnfJAvQMq): update_context_2_and_transfer_to_3 *****\nArguments: \n{}\n***************************************************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Tool_Execution\n\n\n>>>>>>>> EXECUTING FUNCTION update_context_2_and_transfer_to_3...\nTool_Execution (to chat_manager):\n\n***** Response from calling tool (call_Yf5DTGaaYkA726ubnfJAvQMq) *****\n\n**********************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_3\n\nAgent_3 (to chat_manager):\n\n***** Suggested tool call (call_jqZNHuMtQYeNh5Mq4pV2uwAj): transfer_to_Agent_4 *****\nArguments: \n{}\n************************************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Tool_Execution\n\n\n>>>>>>>> EXECUTING FUNCTION transfer_to_Agent_4...\nTool_Execution (to chat_manager):\n\n***** Response from calling tool (call_jqZNHuMtQYeNh5Mq4pV2uwAj) *****\nSwarmAgent --\x3e Agent_4\n**********************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_4\n\nAgent_4 (to chat_manager):\n\n***** Suggested tool call (call_KeNGv98klvDZsrAX10Ou3I71): get_random_number *****\nArguments: \n{}\n**********************************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Tool_Execution\n\n\n>>>>>>>> EXECUTING FUNCTION get_random_number...\nTool_Execution (to chat_manager):\n\n***** Response from calling tool (call_KeNGv98klvDZsrAX10Ou3I71) *****\n27\n**********************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_4\n\nAgent_4 (to chat_manager):\n\nThe random number generated is 27.\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_5\n\nAgent_5 (to chat_manager):\n\n***** Suggested tool call (call_MlSGNNktah3m3QGssWBEzxCe): update_context_3_with_random_number *****\nArguments: \n{"random_number":27}\n****************************************************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Tool_Execution\n\n\n>>>>>>>> EXECUTING FUNCTION update_context_3_with_random_number...\nTool_Execution (to chat_manager):\n\n***** Response from calling tool (call_MlSGNNktah3m3QGssWBEzxCe) *****\n\n**********************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_5\n\nAgent_5 (to chat_manager):\n\nThe random number 27 has been successfully updated in context 3.\n\n--------------------------------------------------------------------------------\n'})}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:"print(context_variables)\n"})}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-text",children:"{'1': True, '2': True, '3': 27}\n"})}),"\n",(0,a.jsx)(e.h3,{id:"demo-with-user-agent",children:"Demo with User Agent"}),"\n",(0,a.jsxs)(e.p,{children:["We pass in a user agent to the swarm chat to accept user inputs. With\n",(0,a.jsx)(e.code,{children:"agent_6"}),", we register an ",(0,a.jsx)(e.code,{children:"AFTER_WORK"})," handoff to revert to the user\nagent when no tool calls are suggested."]}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-python",children:'from autogen import UserProxyAgent\n\nuser_agent = UserProxyAgent(name="User", code_execution_config=False)\n\nagent_6 = SwarmAgent(\n    name="Agent_6",\n    system_message="You are Agent 6. Your job is to tell jokes.",\n    llm_config=llm_config,\n)\n\nagent_7 = SwarmAgent(\n    name="Agent_7",\n    system_message="You are Agent 7, explain the joke.",\n    llm_config=llm_config,\n)\n\nagent_6.register_hand_off(\n    [\n        ON_CONDITION(\n            agent_7, "Used to transfer to Agent 7. Don\'t call this function, unless the user explicitly tells you to."\n        ),\n        AFTER_WORK(AfterWorkOption.REVERT_TO_USER),\n    ]\n)\n\nchat_result, _, _ = initiate_swarm_chat(\n    init_agent=agent_6,\n    agents=[agent_6, agent_7],\n    user_agent=user_agent,\n    messages="start",\n)\n'})}),"\n",(0,a.jsx)(e.pre,{children:(0,a.jsx)(e.code,{className:"language-text",children:'User (to chat_manager):\n\nstart\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_6\n\nAgent_6 (to chat_manager):\n\nWhy did the scarecrow win an award? \n\nBecause he was outstanding in his field! \n\nWant to hear another one?\n\n--------------------------------------------------------------------------------\n\nNext speaker: User\n\nUser (to chat_manager):\n\nyes\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_6\n\nAgent_6 (to chat_manager):\n\nWhy don\'t skeletons fight each other?\n\nThey don\'t have the guts! \n\nHow about another?\n\n--------------------------------------------------------------------------------\n\nNext speaker: User\n\nUser (to chat_manager):\n\ntransfer\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_6\n\nAgent_6 (to chat_manager):\n\n***** Suggested tool call (call_gQ9leFamxgzQp8ZVQB8rUH73): transfer_to_Agent_7 *****\nArguments: \n{}\n************************************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Tool_Execution\n\n\n>>>>>>>> EXECUTING FUNCTION transfer_to_Agent_7...\nTool_Execution (to chat_manager):\n\n***** Response from calling tool (call_gQ9leFamxgzQp8ZVQB8rUH73) *****\nSwarmAgent --\x3e Agent_7\n**********************************************************************\n\n--------------------------------------------------------------------------------\n\nNext speaker: Agent_7\n\nAgent_7 (to chat_manager):\n\nThe joke about the scarecrow winning an award is a play on words. It utilizes the term "outstanding," which can mean both exceptionally good (in the context of the scarecrow\'s performance) and literally being "standing out" in a field (where scarecrows are placed). So, the double meaning creates a pun that makes the joke humorous. \n\nThe skeleton joke works similarly. When it says skeletons "don\'t have the guts," it plays on the literal fact that skeletons don\'t have internal organs (guts), and metaphorically, "having guts" means having courage. The humor comes from this clever wordplay.\n\n--------------------------------------------------------------------------------\n'})})]})}function g(n={}){const{wrapper:e}={...(0,s.a)(),...n.components};return e?(0,a.jsx)(e,{...n,children:(0,a.jsx)(h,{...n})}):h(n)}},11151:(n,e,t)=>{t.d(e,{Z:()=>i,a:()=>r});var a=t(67294);const s={},o=a.createContext(s);function r(n){const e=a.useContext(o);return a.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function i(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(s):n.components||s:r(n.components),a.createElement(o.Provider,{value:e},n.children)}}}]);