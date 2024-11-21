"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[51132],{98364:(n,e,t)=>{t.r(e),t.d(e,{assets:()=>i,contentTitle:()=>a,default:()=>h,frontMatter:()=>c,metadata:()=>s,toc:()=>l});var o=t(85893),r=t(11151);const c={custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_function_call_currency_calculator.ipynb",description:"Learn how to register function calls using AssistantAgent and UserProxyAgent.",source_notebook:"/notebook/agentchat_function_call_currency_calculator.ipynb",tags:["tool/function"],title:"Currency Calculator: Task Solving with Provided Tools as Functions"},a="Currency Calculator: Task Solving with Provided Tools as Functions",s={id:"notebooks/agentchat_function_call_currency_calculator",title:"Currency Calculator: Task Solving with Provided Tools as Functions",description:"Learn how to register function calls using AssistantAgent and UserProxyAgent.",source:"@site/docs/notebooks/agentchat_function_call_currency_calculator.mdx",sourceDirName:"notebooks",slug:"/notebooks/agentchat_function_call_currency_calculator",permalink:"/ag2/docs/notebooks/agentchat_function_call_currency_calculator",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_function_call_currency_calculator.ipynb",tags:[{label:"tool/function",permalink:"/ag2/docs/tags/tool-function"}],version:"current",frontMatter:{custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_function_call_currency_calculator.ipynb",description:"Learn how to register function calls using AssistantAgent and UserProxyAgent.",source_notebook:"/notebook/agentchat_function_call_currency_calculator.ipynb",tags:["tool/function"],title:"Currency Calculator: Task Solving with Provided Tools as Functions"},sidebar:"notebooksSidebar",previous:{title:"Writing a software application using function calls",permalink:"/ag2/docs/notebooks/agentchat_function_call_code_writing"},next:{title:"Groupchat with Llamaindex agents",permalink:"/ag2/docs/notebooks/agentchat_group_chat_with_llamaindex_agents"}},i={},l=[{value:"Requirements",id:"requirements",level:2},{value:"Set your API Endpoint",id:"set-your-api-endpoint",level:2},{value:"Making Function Calls",id:"making-function-calls",level:2},{value:"Pydantic models",id:"pydantic-models",level:3}];function u(n){const e={a:"a",code:"code",h1:"h1",h2:"h2",h3:"h3",img:"img",li:"li",p:"p",pre:"pre",ul:"ul",...(0,r.a)(),...n.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(e.h1,{id:"currency-calculator-task-solving-with-provided-tools-as-functions",children:"Currency Calculator: Task Solving with Provided Tools as Functions"}),"\n",(0,o.jsxs)(e.p,{children:[(0,o.jsx)(e.a,{href:"https://colab.research.google.com/github/ag2ai/ag2/blob/main/notebook/agentchat_function_call_currency_calculator.ipynb",children:(0,o.jsx)(e.img,{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Open In Colab"})}),"\n",(0,o.jsx)(e.a,{href:"https://github.com/ag2ai/ag2/blob/main/notebook/agentchat_function_call_currency_calculator.ipynb",children:(0,o.jsx)(e.img,{src:"https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github",alt:"Open on GitHub"})})]}),"\n",(0,o.jsxs)(e.p,{children:["AutoGen offers conversable agents powered by LLM, tool, or human, which\ncan be used to perform tasks collectively via automated chat. This\nframework allows tool use and human participation through multi-agent\nconversation. Please find documentation about this feature\n",(0,o.jsx)(e.a,{href:"https://ag2ai.github.io/ag2/docs/Use-Cases/agent_chat",children:"here"}),"."]}),"\n",(0,o.jsxs)(e.p,{children:["In this notebook, we demonstrate how to use ",(0,o.jsx)(e.code,{children:"AssistantAgent"})," and\n",(0,o.jsx)(e.code,{children:"UserProxyAgent"})," to make function calls with the new feature of OpenAI\nmodels (in model version 0613). A specified prompt and function configs\nmust be passed to ",(0,o.jsx)(e.code,{children:"AssistantAgent"})," to initialize the agent. The\ncorresponding functions must be passed to ",(0,o.jsx)(e.code,{children:"UserProxyAgent"}),", which will\nexecute any function calls made by ",(0,o.jsx)(e.code,{children:"AssistantAgent"}),". Besides this\nrequirement of matching descriptions with functions, we recommend\nchecking the system message in the ",(0,o.jsx)(e.code,{children:"AssistantAgent"})," to ensure the\ninstructions align with the function call descriptions."]}),"\n",(0,o.jsx)(e.h2,{id:"requirements",children:"Requirements"}),"\n",(0,o.jsxs)(e.p,{children:["AutoGen requires ",(0,o.jsx)(e.code,{children:"Python>=3.8"}),". To run this notebook example, please\ninstall ",(0,o.jsx)(e.code,{children:"pyautogen"}),":"]}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-bash",children:"pip install pyautogen\n"})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'# %pip install "pyautogen>=0.2.3"\n'})}),"\n",(0,o.jsx)(e.h2,{id:"set-your-api-endpoint",children:"Set your API Endpoint"}),"\n",(0,o.jsxs)(e.p,{children:["The\n",(0,o.jsx)(e.a,{href:"https://ag2ai.github.io/ag2/docs/reference/oai/openai_utils#config_list_from_json",children:(0,o.jsx)(e.code,{children:"config_list_from_json"})}),"\nfunction loads a list of configurations from an environment variable or\na json file."]}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'from typing import Literal\n\nfrom pydantic import BaseModel, Field\nfrom typing_extensions import Annotated\n\nimport autogen\nfrom autogen.cache import Cache\n\nconfig_list = autogen.config_list_from_json(\n    "OAI_CONFIG_LIST",\n    filter_dict={"tags": ["3.5-tool"]},  # comment out to get all\n)\n'})}),"\n",(0,o.jsx)(e.p,{children:"It first looks for environment variable \u201cOAI_CONFIG_LIST\u201d which needs to\nbe a valid json string. If that variable is not found, it then looks for\na json file named \u201cOAI_CONFIG_LIST\u201d. It filters the configs by tags (you\ncan filter by other keys as well). Only the configs with matching tags\nare kept in the list based on the filter condition."}),"\n",(0,o.jsx)(e.p,{children:"The config list looks like the following:"}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:"config_list = [\n    {\n        'model': 'gpt-3.5-turbo',\n        'api_key': '<your OpenAI API key here>',\n        'tags': ['tool', '3.5-tool'],\n    },\n    {\n        'model': 'gpt-3.5-turbo',\n        'api_key': '<your Azure OpenAI API key here>',\n        'base_url': '<your Azure OpenAI API base here>',\n        'api_type': 'azure',\n        'api_version': '2024-02-01',\n        'tags': ['tool', '3.5-tool'],\n    },\n    {\n        'model': 'gpt-3.5-turbo-16k',\n        'api_key': '<your Azure OpenAI API key here>',\n        'base_url': '<your Azure OpenAI API base here>',\n        'api_type': 'azure',\n        'api_version': '2024-02-01',\n        'tags': ['tool', '3.5-tool'],\n    },\n]\n"})}),"\n",(0,o.jsxs)(e.p,{children:["You can set the value of config_list in any way you prefer. Please refer\nto this\n",(0,o.jsx)(e.a,{href:"https://github.com/microsoft/autogen/blob/main/website/docs/topics/llm_configuration.ipynb",children:"notebook"}),"\nfor full code examples of the different methods."]}),"\n",(0,o.jsx)(e.h2,{id:"making-function-calls",children:"Making Function Calls"}),"\n",(0,o.jsxs)(e.p,{children:["In this example, we demonstrate function call execution with\n",(0,o.jsx)(e.code,{children:"AssistantAgent"})," and ",(0,o.jsx)(e.code,{children:"UserProxyAgent"}),". With the default system prompt of\n",(0,o.jsx)(e.code,{children:"AssistantAgent"}),", we allow the LLM assistant to perform tasks with code,\nand the ",(0,o.jsx)(e.code,{children:"UserProxyAgent"})," would extract code blocks from the LLM response\nand execute them. With the new \u201cfunction_call\u201d feature, we define\nfunctions and specify the description of the function in the OpenAI\nconfig for the ",(0,o.jsx)(e.code,{children:"AssistantAgent"}),". Then we register the functions in\n",(0,o.jsx)(e.code,{children:"UserProxyAgent"}),"."]}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'llm_config = {\n    "config_list": config_list,\n    "timeout": 120,\n}\n\nchatbot = autogen.AssistantAgent(\n    name="chatbot",\n    system_message="For currency exchange tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",\n    llm_config=llm_config,\n)\n\n# create a UserProxyAgent instance named "user_proxy"\nuser_proxy = autogen.UserProxyAgent(\n    name="user_proxy",\n    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),\n    human_input_mode="NEVER",\n    max_consecutive_auto_reply=10,\n)\n\n\nCurrencySymbol = Literal["USD", "EUR"]\n\n\ndef exchange_rate(base_currency: CurrencySymbol, quote_currency: CurrencySymbol) -> float:\n    if base_currency == quote_currency:\n        return 1.0\n    elif base_currency == "USD" and quote_currency == "EUR":\n        return 1 / 1.1\n    elif base_currency == "EUR" and quote_currency == "USD":\n        return 1.1\n    else:\n        raise ValueError(f"Unknown currencies {base_currency}, {quote_currency}")\n\n\n@user_proxy.register_for_execution()\n@chatbot.register_for_llm(description="Currency exchange calculator.")\ndef currency_calculator(\n    base_amount: Annotated[float, "Amount of currency in base_currency"],\n    base_currency: Annotated[CurrencySymbol, "Base currency"] = "USD",\n    quote_currency: Annotated[CurrencySymbol, "Quote currency"] = "EUR",\n) -> str:\n    quote_amount = exchange_rate(base_currency, quote_currency) * base_amount\n    return f"{quote_amount} {quote_currency}"\n'})}),"\n",(0,o.jsxs)(e.p,{children:["The decorator ",(0,o.jsx)(e.code,{children:"@chatbot.register_for_llm()"})," reads the annotated\nsignature of the function ",(0,o.jsx)(e.code,{children:"currency_calculator"})," and generates the\nfollowing JSON schema used by OpenAI API to suggest calling the\nfunction. We can check the JSON schema generated as follows:"]}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'chatbot.llm_config["tools"]\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-text",children:"[{'type': 'function',\n  'function': {'description': 'Currency exchange calculator.',\n   'name': 'currency_calculator',\n   'parameters': {'type': 'object',\n    'properties': {'base_amount': {'type': 'number',\n      'description': 'Amount of currency in base_currency'},\n     'base_currency': {'enum': ['USD', 'EUR'],\n      'type': 'string',\n      'default': 'USD',\n      'description': 'Base currency'},\n     'quote_currency': {'enum': ['USD', 'EUR'],\n      'type': 'string',\n      'default': 'EUR',\n      'description': 'Quote currency'}},\n    'required': ['base_amount']}}}]\n"})}),"\n",(0,o.jsxs)(e.p,{children:["The decorator ",(0,o.jsx)(e.code,{children:"@user_proxy.register_for_execution()"})," maps the name of\nthe function to be proposed by OpenAI API to the actual implementation.\nThe function mapped is wrapped since we also automatically handle\nserialization of the output of function as follows:"]}),"\n",(0,o.jsxs)(e.ul,{children:["\n",(0,o.jsxs)(e.li,{children:["\n",(0,o.jsx)(e.p,{children:"string are untouched, and"}),"\n"]}),"\n",(0,o.jsxs)(e.li,{children:["\n",(0,o.jsx)(e.p,{children:"objects of the Pydantic BaseModel type are serialized to JSON."}),"\n"]}),"\n"]}),"\n",(0,o.jsxs)(e.p,{children:["We can check the correctness of function map by using ",(0,o.jsx)(e.code,{children:"._origin"}),"\nproperty of the wrapped function as follows:"]}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'assert user_proxy.function_map["currency_calculator"]._origin == currency_calculator\n'})}),"\n",(0,o.jsx)(e.p,{children:"Finally, we can use this function to accurately calculate exchange\namounts:"}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'with Cache.disk() as cache:\n    # start the conversation\n    res = user_proxy.initiate_chat(\n        chatbot, message="How much is 123.45 USD in EUR?", summary_method="reflection_with_llm", cache=cache\n    )\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-text",children:'user_proxy (to chatbot):\n\nHow much is 123.45 USD in EUR?\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\n***** Suggested tool call (call_9ogJS4d40BT1rXfMn7YJb151): currency_calculator *****\nArguments: \n{\n  "base_amount": 123.45,\n  "base_currency": "USD",\n  "quote_currency": "EUR"\n}\n************************************************************************************\n\n--------------------------------------------------------------------------------\n\n>>>>>>>> EXECUTING FUNCTION currency_calculator...\nuser_proxy (to chatbot):\n\nuser_proxy (to chatbot):\n\n***** Response from calling tool (call_9ogJS4d40BT1rXfMn7YJb151) *****\n112.22727272727272 EUR\n**********************************************************************\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\n123.45 USD is equivalent to 112.23 EUR.\n\n--------------------------------------------------------------------------------\nuser_proxy (to chatbot):\n\n\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\nTERMINATE\n\n--------------------------------------------------------------------------------\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'print("Chat summary:", res.summary)\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-text",children:"Chat summary: 123.45 USD is equivalent to 112.23 EUR.\n"})}),"\n",(0,o.jsx)(e.h3,{id:"pydantic-models",children:"Pydantic models"}),"\n",(0,o.jsx)(e.p,{children:"We can also use Pydantic Base models to rewrite the function as follows:"}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'llm_config = {\n    "config_list": config_list,\n    "timeout": 120,\n}\n\nchatbot = autogen.AssistantAgent(\n    name="chatbot",\n    system_message="For currency exchange tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",\n    llm_config=llm_config,\n)\n\n# create a UserProxyAgent instance named "user_proxy"\nuser_proxy = autogen.UserProxyAgent(\n    name="user_proxy",\n    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),\n    human_input_mode="NEVER",\n    max_consecutive_auto_reply=10,\n)\n\n\nclass Currency(BaseModel):\n    currency: Annotated[CurrencySymbol, Field(..., description="Currency symbol")]\n    amount: Annotated[float, Field(0, description="Amount of currency", ge=0)]\n\n\n# another way to register a function is to use register_function instead of register_for_execution and register_for_llm decorators\ndef currency_calculator(\n    base: Annotated[Currency, "Base currency: amount and currency symbol"],\n    quote_currency: Annotated[CurrencySymbol, "Quote currency symbol"] = "USD",\n) -> Currency:\n    quote_amount = exchange_rate(base.currency, quote_currency) * base.amount\n    return Currency(amount=quote_amount, currency=quote_currency)\n\n\nautogen.agentchat.register_function(\n    currency_calculator,\n    caller=chatbot,\n    executor=user_proxy,\n    description="Currency exchange calculator.",\n)\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'chatbot.llm_config["tools"]\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-text",children:"[{'type': 'function',\n  'function': {'description': 'Currency exchange calculator.',\n   'name': 'currency_calculator',\n   'parameters': {'type': 'object',\n    'properties': {'base': {'properties': {'currency': {'description': 'Currency symbol',\n        'enum': ['USD', 'EUR'],\n        'title': 'Currency',\n        'type': 'string'},\n       'amount': {'default': 0,\n        'description': 'Amount of currency',\n        'minimum': 0.0,\n        'title': 'Amount',\n        'type': 'number'}},\n      'required': ['currency'],\n      'title': 'Currency',\n      'type': 'object',\n      'description': 'Base currency: amount and currency symbol'},\n     'quote_currency': {'enum': ['USD', 'EUR'],\n      'type': 'string',\n      'default': 'USD',\n      'description': 'Quote currency symbol'}},\n    'required': ['base']}}}]\n"})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'with Cache.disk() as cache:\n    # start the conversation\n    res = user_proxy.initiate_chat(\n        chatbot, message="How much is 112.23 Euros in US Dollars?", summary_method="reflection_with_llm", cache=cache\n    )\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-text",children:'user_proxy (to chatbot):\n\nHow much is 112.23 Euros in US Dollars?\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\n***** Suggested tool call (call_BQkSmdFHsrKvmtDWCk0mY5sF): currency_calculator *****\nArguments: \n{\n  "base": {\n    "currency": "EUR",\n    "amount": 112.23\n  },\n  "quote_currency": "USD"\n}\n************************************************************************************\n\n--------------------------------------------------------------------------------\n\n>>>>>>>> EXECUTING FUNCTION currency_calculator...\nuser_proxy (to chatbot):\n\nuser_proxy (to chatbot):\n\n***** Response from calling tool (call_BQkSmdFHsrKvmtDWCk0mY5sF) *****\n{"currency":"USD","amount":123.45300000000002}\n**********************************************************************\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\n112.23 Euros is equivalent to 123.45 US Dollars.\nTERMINATE\n\n--------------------------------------------------------------------------------\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'print("Chat summary:", res.summary)\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-text",children:"Chat summary: 112.23 Euros is equivalent to 123.45 US Dollars.\n"})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'with Cache.disk() as cache:\n    # start the conversation\n    res = user_proxy.initiate_chat(\n        chatbot,\n        message="How much is 123.45 US Dollars in Euros?",\n        cache=cache,\n    )\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-text",children:'user_proxy (to chatbot):\n\nHow much is 123.45 US Dollars in Euros?\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\n***** Suggested tool call (call_Xxol42xTswZHGX60OjvIQRG1): currency_calculator *****\nArguments: \n{\n  "base": {\n    "currency": "USD",\n    "amount": 123.45\n  },\n  "quote_currency": "EUR"\n}\n************************************************************************************\n\n--------------------------------------------------------------------------------\n\n>>>>>>>> EXECUTING FUNCTION currency_calculator...\nuser_proxy (to chatbot):\n\nuser_proxy (to chatbot):\n\n***** Response from calling tool (call_Xxol42xTswZHGX60OjvIQRG1) *****\n{"currency":"EUR","amount":112.22727272727272}\n**********************************************************************\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\n123.45 US Dollars is equivalent to 112.23 Euros.\n\n--------------------------------------------------------------------------------\nuser_proxy (to chatbot):\n\n\n\n--------------------------------------------------------------------------------\nchatbot (to user_proxy):\n\nTERMINATE\n\n--------------------------------------------------------------------------------\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-python",children:'print("Chat history:", res.chat_history)\n'})}),"\n",(0,o.jsx)(e.pre,{children:(0,o.jsx)(e.code,{className:"language-text",children:"Chat history: [{'content': 'How much is 123.45 US Dollars in Euros?', 'role': 'assistant'}, {'tool_calls': [{'id': 'call_Xxol42xTswZHGX60OjvIQRG1', 'function': {'arguments': '{\\n  \"base\": {\\n    \"currency\": \"USD\",\\n    \"amount\": 123.45\\n  },\\n  \"quote_currency\": \"EUR\"\\n}', 'name': 'currency_calculator'}, 'type': 'function'}], 'content': None, 'role': 'assistant'}, {'content': '{\"currency\":\"EUR\",\"amount\":112.22727272727272}', 'tool_responses': [{'tool_call_id': 'call_Xxol42xTswZHGX60OjvIQRG1', 'role': 'tool', 'content': '{\"currency\":\"EUR\",\"amount\":112.22727272727272}'}], 'role': 'tool'}, {'content': '123.45 US Dollars is equivalent to 112.23 Euros.', 'role': 'user'}, {'content': '', 'role': 'assistant'}, {'content': 'TERMINATE', 'role': 'user'}]\n"})})]})}function h(n={}){const{wrapper:e}={...(0,r.a)(),...n.components};return e?(0,o.jsx)(e,{...n,children:(0,o.jsx)(u,{...n})}):u(n)}},11151:(n,e,t)=>{t.d(e,{Z:()=>s,a:()=>a});var o=t(67294);const r={},c=o.createContext(r);function a(n){const e=o.useContext(c);return o.useMemo((function(){return"function"==typeof n?n(e):{...e,...n}}),[e,n])}function s(n){let e;return e=n.disableParentContext?"function"==typeof n.components?n.components(r):n.components||r:a(n.components),o.createElement(c.Provider,{value:e},n.children)}}}]);