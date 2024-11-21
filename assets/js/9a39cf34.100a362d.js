"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[28096],{88488:(e,c,n)=>{n.r(c),n.d(c,{assets:()=>d,contentTitle:()=>i,default:()=>u,frontMatter:()=>o,metadata:()=>s,toc:()=>a});var r=n(85893),t=n(11151);const o={sidebar_label:"factory",title:"coding.factory"},i=void 0,s={id:"reference/coding/factory",title:"coding.factory",description:"CodeExecutorFactory",source:"@site/docs/reference/coding/factory.md",sourceDirName:"reference/coding",slug:"/reference/coding/factory",permalink:"/ag2/docs/reference/coding/factory",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/website/docs/reference/coding/factory.md",tags:[],version:"current",frontMatter:{sidebar_label:"factory",title:"coding.factory"},sidebar:"referenceSideBar",previous:{title:"docker_commandline_code_executor",permalink:"/ag2/docs/reference/coding/docker_commandline_code_executor"},next:{title:"func_with_reqs",permalink:"/ag2/docs/reference/coding/func_with_reqs"}},d={},a=[{value:"CodeExecutorFactory",id:"codeexecutorfactory",level:2},{value:"create",id:"create",level:3}];function l(e){const c={code:"code",em:"em",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,t.a)(),...e.components};return(0,r.jsxs)(r.Fragment,{children:[(0,r.jsx)(c.h2,{id:"codeexecutorfactory",children:"CodeExecutorFactory"}),"\n",(0,r.jsx)(c.pre,{children:(0,r.jsx)(c.code,{className:"language-python",children:"class CodeExecutorFactory()\n"})}),"\n",(0,r.jsx)(c.p,{children:"(Experimental) A factory class for creating code executors."}),"\n",(0,r.jsx)(c.h3,{id:"create",children:"create"}),"\n",(0,r.jsx)(c.pre,{children:(0,r.jsx)(c.code,{className:"language-python",children:"@staticmethod\ndef create(code_execution_config: CodeExecutionConfig) -> CodeExecutor\n"})}),"\n",(0,r.jsx)(c.p,{children:"(Experimental) Get a code executor based on the code execution config."}),"\n",(0,r.jsxs)(c.p,{children:[(0,r.jsx)(c.strong,{children:"Arguments"}),":"]}),"\n",(0,r.jsxs)(c.ul,{children:["\n",(0,r.jsxs)(c.li,{children:[(0,r.jsx)(c.code,{children:"code_execution_config"})," ",(0,r.jsx)(c.em,{children:"Dict"}),' - The code execution config,\nwhich is a dictionary that must contain the key "executor".\nThe value of the key "executor" can be either a string\nor an instance of CodeExecutor, in which case the code\nexecutor is returned directly.']}),"\n"]}),"\n",(0,r.jsxs)(c.p,{children:[(0,r.jsx)(c.strong,{children:"Returns"}),":"]}),"\n",(0,r.jsxs)(c.ul,{children:["\n",(0,r.jsxs)(c.li,{children:[(0,r.jsx)(c.code,{children:"CodeExecutor"})," - The code executor."]}),"\n"]}),"\n",(0,r.jsxs)(c.p,{children:[(0,r.jsx)(c.strong,{children:"Raises"}),":"]}),"\n",(0,r.jsxs)(c.ul,{children:["\n",(0,r.jsxs)(c.li,{children:[(0,r.jsx)(c.code,{children:"ValueError"})," - If the code executor is unknown or not specified."]}),"\n"]})]})}function u(e={}){const{wrapper:c}={...(0,t.a)(),...e.components};return c?(0,r.jsx)(c,{...e,children:(0,r.jsx)(l,{...e})}):l(e)}},11151:(e,c,n)=>{n.d(c,{Z:()=>s,a:()=>i});var r=n(67294);const t={},o=r.createContext(t);function i(e){const c=r.useContext(o);return r.useMemo((function(){return"function"==typeof e?e(c):{...c,...e}}),[c,e])}function s(e){let c;return c=e.disableParentContext?"function"==typeof e.components?e.components(t):e.components||t:i(e.components),r.createElement(o.Provider,{value:c},e.children)}}}]);