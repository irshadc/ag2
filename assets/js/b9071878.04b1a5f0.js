"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[81383],{8408:(e,s,n)=>{n.r(s),n.d(s,{assets:()=>i,contentTitle:()=>o,default:()=>h,frontMatter:()=>a,metadata:()=>c,toc:()=>l});var t=n(85893),r=n(11151);const a={sidebar_label:"bedrock",title:"oai.bedrock"},o=void 0,c={id:"reference/oai/bedrock",title:"oai.bedrock",description:"Create a compatible client for the Amazon Bedrock Converse API.",source:"@site/docs/reference/oai/bedrock.md",sourceDirName:"reference/oai",slug:"/reference/oai/bedrock",permalink:"/ag2/docs/reference/oai/bedrock",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/website/docs/reference/oai/bedrock.md",tags:[],version:"current",frontMatter:{sidebar_label:"bedrock",title:"oai.bedrock"},sidebar:"referenceSideBar",previous:{title:"anthropic",permalink:"/ag2/docs/reference/oai/anthropic"},next:{title:"cerebras",permalink:"/ag2/docs/reference/oai/cerebras"}},i={},l=[{value:"BedrockClient",id:"bedrockclient",level:2},{value:"__init__",id:"__init__",level:3},{value:"message_retrieval",id:"message_retrieval",level:3},{value:"parse_custom_params",id:"parse_custom_params",level:3},{value:"parse_params",id:"parse_params",level:3},{value:"create",id:"create",level:3},{value:"cost",id:"cost",level:3},{value:"get_usage",id:"get_usage",level:3},{value:"extract_system_messages",id:"extract_system_messages",level:3},{value:"oai_messages_to_bedrock_messages",id:"oai_messages_to_bedrock_messages",level:3},{value:"parse_image",id:"parse_image",level:3},{value:"format_tool_calls",id:"format_tool_calls",level:3},{value:"convert_stop_reason_to_finish_reason",id:"convert_stop_reason_to_finish_reason",level:3},{value:"calculate_cost",id:"calculate_cost",level:3}];function d(e){const s={a:"a",code:"code",em:"em",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.a)(),...e.components};return(0,t.jsxs)(t.Fragment,{children:[(0,t.jsx)(s.p,{children:"Create a compatible client for the Amazon Bedrock Converse API."}),"\n",(0,t.jsxs)(s.p,{children:["Example usage:\nInstall the ",(0,t.jsx)(s.code,{children:"boto3"})," package by running ",(0,t.jsx)(s.code,{children:"pip install --upgrade boto3"}),"."]}),"\n",(0,t.jsxs)(s.ul,{children:["\n",(0,t.jsx)(s.li,{children:(0,t.jsx)(s.a,{href:"https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html",children:"https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html"})}),"\n"]}),"\n",(0,t.jsx)(s.p,{children:"import autogen"}),"\n",(0,t.jsx)(s.p,{children:'config_list = [\n{\n"api_type": "bedrock",\n"model": "meta.llama3-1-8b-instruct-v1:0",\n"aws_region": "us-west-2",\n"aws_access_key": "",\n"aws_secret_key": "",\n"price" : [0.003, 0.015]\n}\n]'}),"\n",(0,t.jsx)(s.p,{children:'assistant = autogen.AssistantAgent("assistant", llm_config={"config_list": config_list})'}),"\n",(0,t.jsx)(s.h2,{id:"bedrockclient",children:"BedrockClient"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"class BedrockClient()\n"})}),"\n",(0,t.jsx)(s.p,{children:"Client for Amazon's Bedrock Converse API."}),"\n",(0,t.jsx)(s.h3,{id:"__init__",children:"__init__"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def __init__(**kwargs: Any)\n"})}),"\n",(0,t.jsx)(s.p,{children:"Initialises BedrockClient for Amazon's Bedrock Converse API"}),"\n",(0,t.jsx)(s.h3,{id:"message_retrieval",children:"message_retrieval"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def message_retrieval(response)\n"})}),"\n",(0,t.jsx)(s.p,{children:"Retrieve the messages from the response."}),"\n",(0,t.jsx)(s.h3,{id:"parse_custom_params",children:"parse_custom_params"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def parse_custom_params(params: Dict[str, Any])\n"})}),"\n",(0,t.jsx)(s.p,{children:"Parses custom parameters for logic in this client class"}),"\n",(0,t.jsx)(s.h3,{id:"parse_params",children:"parse_params"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def parse_params(\n        params: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]\n"})}),"\n",(0,t.jsx)(s.p,{children:"Loads the valid parameters required to invoke Bedrock Converse\nReturns a tuple of (base_params, additional_params)"}),"\n",(0,t.jsx)(s.h3,{id:"create",children:"create"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def create(params)\n"})}),"\n",(0,t.jsx)(s.p,{children:"Run Amazon Bedrock inference and return AutoGen response"}),"\n",(0,t.jsx)(s.h3,{id:"cost",children:"cost"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def cost(response: ChatCompletion) -> float\n"})}),"\n",(0,t.jsx)(s.p,{children:"Calculate the cost of the response."}),"\n",(0,t.jsx)(s.h3,{id:"get_usage",children:"get_usage"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"@staticmethod\ndef get_usage(response) -> Dict\n"})}),"\n",(0,t.jsx)(s.p,{children:"Get the usage of tokens and their cost information."}),"\n",(0,t.jsx)(s.h3,{id:"extract_system_messages",children:"extract_system_messages"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def extract_system_messages(messages: List[dict]) -> List\n"})}),"\n",(0,t.jsx)(s.p,{children:"Extract the system messages from the list of messages."}),"\n",(0,t.jsxs)(s.p,{children:[(0,t.jsx)(s.strong,{children:"Arguments"}),":"]}),"\n",(0,t.jsxs)(s.ul,{children:["\n",(0,t.jsxs)(s.li,{children:[(0,t.jsx)(s.code,{children:"messages"})," ",(0,t.jsx)(s.em,{children:"list[dict]"})," - List of messages."]}),"\n"]}),"\n",(0,t.jsxs)(s.p,{children:[(0,t.jsx)(s.strong,{children:"Returns"}),":"]}),"\n",(0,t.jsxs)(s.ul,{children:["\n",(0,t.jsxs)(s.li,{children:[(0,t.jsx)(s.code,{children:"List[SystemMessage]"})," - List of System messages."]}),"\n"]}),"\n",(0,t.jsx)(s.h3,{id:"oai_messages_to_bedrock_messages",children:"oai_messages_to_bedrock_messages"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def oai_messages_to_bedrock_messages(\n        messages: List[Dict[str, Any]], has_tools: bool,\n        supports_system_prompts: bool) -> List[Dict]\n"})}),"\n",(0,t.jsx)(s.p,{children:'Convert messages from OAI format to Bedrock format.\nWe correct for any specific role orders and types, etc.\nAWS Bedrock requires messages to alternate between user and assistant roles. This function ensures that the messages\nare in the correct order and format for Bedrock by inserting "Please continue" messages as needed.\nThis is the same method as the one in the Autogen Anthropic client'}),"\n",(0,t.jsx)(s.h3,{id:"parse_image",children:"parse_image"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def parse_image(image_url: str) -> Tuple[bytes, str]\n"})}),"\n",(0,t.jsx)(s.p,{children:"Try to get the raw data from an image url."}),"\n",(0,t.jsxs)(s.p,{children:["Ref: ",(0,t.jsx)(s.a,{href:"https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html",children:"https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageSource.html"}),"\nreturns a tuple of (Image Data, Content Type)"]}),"\n",(0,t.jsx)(s.h3,{id:"format_tool_calls",children:"format_tool_calls"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def format_tool_calls(content)\n"})}),"\n",(0,t.jsx)(s.p,{children:"Converts Converse API response tool calls to AutoGen format"}),"\n",(0,t.jsx)(s.h3,{id:"convert_stop_reason_to_finish_reason",children:"convert_stop_reason_to_finish_reason"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:'def convert_stop_reason_to_finish_reason(\n    stop_reason: str\n) -> Literal["stop", "length", "tool_calls", "content_filter"]\n'})}),"\n",(0,t.jsx)(s.p,{children:"Converts Bedrock finish reasons to our finish reasons, according to OpenAI:"}),"\n",(0,t.jsxs)(s.ul,{children:["\n",(0,t.jsx)(s.li,{children:"stop: if the model hit a natural stop point or a provided stop sequence,"}),"\n",(0,t.jsx)(s.li,{children:"length: if the maximum number of tokens specified in the request was reached,"}),"\n",(0,t.jsx)(s.li,{children:"content_filter: if content was omitted due to a flag from our content filters,"}),"\n",(0,t.jsx)(s.li,{children:"tool_calls: if the model called a tool"}),"\n"]}),"\n",(0,t.jsx)(s.h3,{id:"calculate_cost",children:"calculate_cost"}),"\n",(0,t.jsx)(s.pre,{children:(0,t.jsx)(s.code,{className:"language-python",children:"def calculate_cost(input_tokens: int, output_tokens: int,\n                   model_id: str) -> float\n"})}),"\n",(0,t.jsx)(s.p,{children:"Calculate the cost of the completion using the Bedrock pricing."})]})}function h(e={}){const{wrapper:s}={...(0,r.a)(),...e.components};return s?(0,t.jsx)(s,{...e,children:(0,t.jsx)(d,{...e})}):d(e)}},11151:(e,s,n)=>{n.d(s,{Z:()=>c,a:()=>o});var t=n(67294);const r={},a=t.createContext(r);function o(e){const s=t.useContext(a);return t.useMemo((function(){return"function"==typeof e?e(s):{...s,...e}}),[s,e])}function c(e){let s;return s=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:o(e.components),t.createElement(a.Provider,{value:s},e.children)}}}]);