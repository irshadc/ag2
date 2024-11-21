"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[89202],{16956:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>o,contentTitle:()=>i,default:()=>d,frontMatter:()=>a,metadata:()=>c,toc:()=>l});var s=t(85893),r=t(11151);const a={sidebar_label:"utils",title:"agentchat.utils"},i=void 0,c={id:"reference/agentchat/utils",title:"agentchat.utils",description:"gather\\usage\\summary",source:"@site/docs/reference/agentchat/utils.md",sourceDirName:"reference/agentchat",slug:"/reference/agentchat/utils",permalink:"/ag2/docs/reference/agentchat/utils",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/website/docs/reference/agentchat/utils.md",tags:[],version:"current",frontMatter:{sidebar_label:"utils",title:"agentchat.utils"},sidebar:"referenceSideBar",previous:{title:"user_proxy_agent",permalink:"/ag2/docs/reference/agentchat/user_proxy_agent"},next:{title:"abstract_cache_base",permalink:"/ag2/docs/reference/cache/abstract_cache_base"}},o={},l=[{value:"gather_usage_summary",id:"gather_usage_summary",level:3},{value:"parse_tags_from_content",id:"parse_tags_from_content",level:3}];function h(e){const n={a:"a",audio:"audio",code:"code",em:"em",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,r.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.h3,{id:"gather_usage_summary",children:"gather_usage_summary"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def gather_usage_summary(\n        agents: List[Agent]) -> Dict[Dict[str, Dict], Dict[str, Dict]]\n"})}),"\n",(0,s.jsx)(n.p,{children:"Gather usage summary from all agents."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"agents"})," - (list): List of agents."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"dictionary"})," - A dictionary containing two keys:\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsx)(n.li,{children:'"usage_including_cached_inference": Cost information on the total usage, including the tokens in cached inference.'}),"\n",(0,s.jsx)(n.li,{children:'"usage_excluding_cached_inference": Cost information on the usage of tokens, excluding the tokens in cache. No larger than "usage_including_cached_inference".'}),"\n"]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Example"}),":"]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'{\n    "usage_including_cached_inference" : {\n        "total_cost": 0.0006090000000000001,\n        "gpt-35-turbo": {\n                "cost": 0.0006090000000000001,\n                "prompt_tokens": 242,\n                "completion_tokens": 123,\n                "total_tokens": 365\n        },\n    },\n\n    "usage_excluding_cached_inference" : {\n        "total_cost": 0.0006090000000000001,\n        "gpt-35-turbo": {\n                "cost": 0.0006090000000000001,\n                "prompt_tokens": 242,\n                "completion_tokens": 123,\n                "total_tokens": 365\n        },\n    }\n}\n'})}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Notes"}),":"]}),"\n",(0,s.jsxs)(n.p,{children:["If none of the agents incurred any cost (not having a client), then the usage_including_cached_inference and usage_excluding_cached_inference will be ",(0,s.jsx)(n.code,{children:"{'total_cost': 0}"}),"."]}),"\n",(0,s.jsx)(n.h3,{id:"parse_tags_from_content",children:"parse_tags_from_content"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def parse_tags_from_content(\n    tag: str,\n    content: Union[str, List[Dict[str,\n                                  Any]]]) -> List[Dict[str, Dict[str, str]]]\n"})}),"\n",(0,s.jsx)(n.p,{children:"Parses HTML style tags from message contents."}),"\n",(0,s.jsx)(n.p,{children:"The parsing is done by looking for patterns in the text that match the format of HTML tags. The tag to be parsed is\nspecified as an argument to the function. The function looks for this tag in the text and extracts its content. The\ncontent of a tag is everything that is inside the tag, between the opening and closing angle brackets. The content\ncan be a single string or a set of attribute-value pairs."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Examples"}),":"]}),"\n",(0,s.jsxs)(n.p,{children:["<img ",(0,s.jsx)(n.a,{href:"http://example.com/image.png%3E",children:"http://example.com/image.png>"}),' -> [{"tag": "img", "attr": {"src": "',(0,s.jsx)(n.a,{href:"http://example.com/image.png",children:"http://example.com/image.png"}),'"}, "match": re.Match}]\n',(0,s.jsx)(n.audio,{text:"Hello I'm a robot",prompt:"whisper",children:" ->"})]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:'[{"tag"'}),' - "audio", "attr": {"text": "Hello I\'m a robot", "prompt": "whisper"}, "match": re.Match}]']}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"tag"})," ",(0,s.jsx)(n.em,{children:"str"})," - The HTML style tag to be parsed."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"content"})," ",(0,s.jsx)(n.em,{children:"Union[str, List[Dict[str, Any]]]"})," - The message content to parse. Can be a string or a list of content\nitems."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsx)(n.p,{children:"List[Dict[str, str]]: A list of dictionaries, where each dictionary represents a parsed tag. Each dictionary\ncontains three key-value pairs: 'type' which is the tag, 'attr' which is a dictionary of the parsed attributes,\nand 'match' which is a regular expression match object."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Raises"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"ValueError"})," - If the content is not a string or a list."]}),"\n"]})]})}function d(e={}){const{wrapper:n}={...(0,r.a)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(h,{...e})}):h(e)}},11151:(e,n,t)=>{t.d(n,{Z:()=>c,a:()=>i});var s=t(67294);const r={},a=s.createContext(r);function i(e){const n=s.useContext(a);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function c(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(r):e.components||r:i(e.components),s.createElement(a.Provider,{value:n},e.children)}}}]);