"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[4828],{65143:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>r,contentTitle:()=>l,default:()=>d,frontMatter:()=>i,metadata:()=>s,toc:()=>c});var o=t(85893),a=t(11151);const i={title:"Use AutoGen for Local LLMs",authors:"jialeliu",tags:["LLM"]},l=void 0,s={permalink:"/ag2/blog/2023/07/14/Local-LLMs",source:"@site/blog/2023-07-14-Local-LLMs/index.md",title:"Use AutoGen for Local LLMs",description:"TL;DR:",date:"2023-07-14T00:00:00.000Z",formattedDate:"July 14, 2023",tags:[{label:"LLM",permalink:"/ag2/blog/tags/llm"}],readingTime:2.13,hasTruncateMarker:!1,authors:[{name:"Jiale Liu",title:"PhD student at Pennsylvania State University",url:"https://github.com/LeoLjl",imageURL:"https://github.com/leoljl.png",key:"jialeliu"}],frontMatter:{title:"Use AutoGen for Local LLMs",authors:"jialeliu",tags:["LLM"]},unlisted:!1,prevItem:{title:"Retrieval-Augmented Generation (RAG) Applications with AutoGen",permalink:"/ag2/blog/2023/10/18/RetrieveChat"},nextItem:{title:"MathChat - An Conversational Framework to Solve Math Problems",permalink:"/ag2/blog/2023/06/28/MathChat"}},r={authorsImageUrls:[void 0]},c=[{value:"Preparations",id:"preparations",level:2},{value:"Clone FastChat",id:"clone-fastchat",level:3},{value:"Download checkpoint",id:"download-checkpoint",level:3},{value:"Initiate server",id:"initiate-server",level:2},{value:"Interact with model using <code>oai.Completion</code> (requires openai&lt;1)",id:"interact-with-model-using-oaicompletion-requires-openai1",level:2},{value:"interacting with multiple local LLMs",id:"interacting-with-multiple-local-llms",level:2},{value:"For Further Reading",id:"for-further-reading",level:2}];function h(e){const n={a:"a",code:"code",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,a.a)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsxs)(n.p,{children:[(0,o.jsx)(n.strong,{children:"TL;DR:"}),"\nWe demonstrate how to use autogen for local LLM application. As an example, we will initiate an endpoint using ",(0,o.jsx)(n.a,{href:"https://github.com/lm-sys/FastChat",children:"FastChat"})," and perform inference on ",(0,o.jsx)(n.a,{href:"https://github.com/THUDM/ChatGLM2-6B",children:"ChatGLMv2-6b"}),"."]}),"\n",(0,o.jsx)(n.h2,{id:"preparations",children:"Preparations"}),"\n",(0,o.jsx)(n.h3,{id:"clone-fastchat",children:"Clone FastChat"}),"\n",(0,o.jsx)(n.p,{children:"FastChat provides OpenAI-compatible APIs for its supported models, so you can use FastChat as a local drop-in replacement for OpenAI APIs. However, its code needs minor modification in order to function properly."}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"git clone https://github.com/lm-sys/FastChat.git\ncd FastChat\n"})}),"\n",(0,o.jsx)(n.h3,{id:"download-checkpoint",children:"Download checkpoint"}),"\n",(0,o.jsx)(n.p,{children:"ChatGLM-6B is an open bilingual language model based on General Language Model (GLM) framework, with 6.2 billion parameters. ChatGLM2-6B is its second-generation version."}),"\n",(0,o.jsxs)(n.p,{children:["Before downloading from HuggingFace Hub, you need to have Git LFS ",(0,o.jsx)(n.a,{href:"https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage",children:"installed"}),"."]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"git clone https://huggingface.co/THUDM/chatglm2-6b\n"})}),"\n",(0,o.jsx)(n.h2,{id:"initiate-server",children:"Initiate server"}),"\n",(0,o.jsx)(n.p,{children:"First, launch the controller"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"python -m fastchat.serve.controller\n"})}),"\n",(0,o.jsx)(n.p,{children:"Then, launch the model worker(s)"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"python -m fastchat.serve.model_worker --model-path chatglm2-6b\n"})}),"\n",(0,o.jsx)(n.p,{children:"Finally, launch the RESTful API server"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"python -m fastchat.serve.openai_api_server --host localhost --port 8000\n"})}),"\n",(0,o.jsxs)(n.p,{children:["Normally this will work. However, if you encounter error like ",(0,o.jsx)(n.a,{href:"https://github.com/lm-sys/FastChat/issues/1641",children:"this"}),", commenting out all the lines containing ",(0,o.jsx)(n.code,{children:"finish_reason"})," in ",(0,o.jsx)(n.code,{children:"fastchat/protocol/api_protocol.py"})," and ",(0,o.jsx)(n.code,{children:"fastchat/protocol/openai_api_protocol.py"})," will fix the problem. The modified code looks like:"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:'class CompletionResponseChoice(BaseModel):\n    index: int\n    text: str\n    logprobs: Optional[int] = None\n    # finish_reason: Optional[Literal["stop", "length"]]\n\nclass CompletionResponseStreamChoice(BaseModel):\n    index: int\n    text: str\n    logprobs: Optional[float] = None\n    # finish_reason: Optional[Literal["stop", "length"]] = None\n'})}),"\n",(0,o.jsxs)(n.h2,{id:"interact-with-model-using-oaicompletion-requires-openai1",children:["Interact with model using ",(0,o.jsx)(n.code,{children:"oai.Completion"})," (requires openai<1)"]}),"\n",(0,o.jsxs)(n.p,{children:["Now the models can be directly accessed through openai-python library as well as ",(0,o.jsx)(n.code,{children:"autogen.oai.Completion"})," and ",(0,o.jsx)(n.code,{children:"autogen.oai.ChatCompletion"}),"."]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:'from autogen import oai\n\n# create a text completion request\nresponse = oai.Completion.create(\n    config_list=[\n        {\n            "model": "chatglm2-6b",\n            "base_url": "http://localhost:8000/v1",\n            "api_type": "openai",\n            "api_key": "NULL", # just a placeholder\n        }\n    ],\n    prompt="Hi",\n)\nprint(response)\n\n# create a chat completion request\nresponse = oai.ChatCompletion.create(\n    config_list=[\n        {\n            "model": "chatglm2-6b",\n            "base_url": "http://localhost:8000/v1",\n            "api_type": "openai",\n            "api_key": "NULL",\n        }\n    ],\n    messages=[{"role": "user", "content": "Hi"}]\n)\nprint(response)\n'})}),"\n",(0,o.jsx)(n.p,{children:"If you would like to switch to different models, download their checkpoints and specify model path when launching model worker(s)."}),"\n",(0,o.jsx)(n.h2,{id:"interacting-with-multiple-local-llms",children:"interacting with multiple local LLMs"}),"\n",(0,o.jsxs)(n.p,{children:["If you would like to interact with multiple LLMs on your local machine, replace the ",(0,o.jsx)(n.code,{children:"model_worker"})," step above with a multi model variant:"]}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-bash",children:"python -m fastchat.serve.multi_model_worker \\\n    --model-path lmsys/vicuna-7b-v1.3 \\\n    --model-names vicuna-7b-v1.3 \\\n    --model-path chatglm2-6b \\\n    --model-names chatglm2-6b\n"})}),"\n",(0,o.jsx)(n.p,{children:"The inference code would be:"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:'from autogen import oai\n\n# create a chat completion request\nresponse = oai.ChatCompletion.create(\n    config_list=[\n        {\n            "model": "chatglm2-6b",\n            "base_url": "http://localhost:8000/v1",\n            "api_type": "openai",\n            "api_key": "NULL",\n        },\n        {\n            "model": "vicuna-7b-v1.3",\n            "base_url": "http://localhost:8000/v1",\n            "api_type": "openai",\n            "api_key": "NULL",\n        }\n    ],\n    messages=[{"role": "user", "content": "Hi"}]\n)\nprint(response)\n'})}),"\n",(0,o.jsx)(n.h2,{id:"for-further-reading",children:"For Further Reading"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.a,{href:"/docs/Getting-Started",children:"Documentation"})," about ",(0,o.jsx)(n.code,{children:"autogen"}),"."]}),"\n",(0,o.jsxs)(n.li,{children:[(0,o.jsx)(n.a,{href:"https://github.com/lm-sys/FastChat",children:"Documentation"})," about FastChat."]}),"\n"]})]})}function d(e={}){const{wrapper:n}={...(0,a.a)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(h,{...e})}):h(e)}},11151:(e,n,t)=>{t.d(n,{Z:()=>s,a:()=>l});var o=t(67294);const a={},i=o.createContext(a);function l(e){const n=o.useContext(i);return o.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function s(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(a):e.components||a:l(e.components),o.createElement(i.Provider,{value:n},e.children)}}}]);