"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[61249],{36964:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>l,contentTitle:()=>a,default:()=>d,frontMatter:()=>o,metadata:()=>r,toc:()=>g});var s=n(85893),i=n(11151);const o={title:"AgentOps, the Best Tool for AutoGen Agent Observability",authors:["areibman","bboynton97"],tags:["LLM","Agent","Observability","AutoGen","AgentOps"]},a="AgentOps, the Best Tool for AutoGen Agent Observability",r={permalink:"/ag2/blog/2024/07/25/AgentOps",source:"@site/blog/2024-07-25-AgentOps/index.mdx",title:"AgentOps, the Best Tool for AutoGen Agent Observability",description:"TL;DR",date:"2024-07-25T00:00:00.000Z",formattedDate:"July 25, 2024",tags:[{label:"LLM",permalink:"/ag2/blog/tags/llm"},{label:"Agent",permalink:"/ag2/blog/tags/agent"},{label:"Observability",permalink:"/ag2/blog/tags/observability"},{label:"AutoGen",permalink:"/ag2/blog/tags/auto-gen"},{label:"AgentOps",permalink:"/ag2/blog/tags/agent-ops"}],readingTime:4.93,hasTruncateMarker:!1,authors:[{name:"Alex Reibman",title:"Co-founder/CEO at AgentOps",url:"https://github.com/areibman",imageURL:"https://github.com/areibman.png",key:"areibman"},{name:"Braelyn Boynton",title:"AI Engineer at AgentOps",url:"https://github.com/bboynton97",imageURL:"https://github.com/bboynton97.png",key:"bboynton97"}],frontMatter:{title:"AgentOps, the Best Tool for AutoGen Agent Observability",authors:["areibman","bboynton97"],tags:["LLM","Agent","Observability","AutoGen","AgentOps"]},unlisted:!1,prevItem:{title:"Unlocking the Power of Agentic Workflows at Nexla with Autogen",permalink:"/ag2/blog/2024/10/23/NOVA"},nextItem:{title:"Enhanced Support for Non-OpenAI Models",permalink:"/ag2/blog/2024/06/24/AltModels-Classes"}},l={authorsImageUrls:[void 0,void 0]},g=[{value:"TL;DR",id:"tldr",level:2},{value:"What is Agent Observability?",id:"what-is-agent-observability",level:2},{value:"Why AgentOps?",id:"why-agentops",level:2},{value:"AgentOps&#39;s Features",id:"agentopss-features",level:2},{value:"Conclusion",id:"conclusion",level:2}];function c(e){const t={a:"a",code:"code",h2:"h2",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)("img",{src:"https://github.com/AgentOps-AI/agentops/blob/main/docs/images/external/autogen/autogen-integration.png?raw=true",alt:"AgentOps and AutoGen",style:{maxWidth:"50%"}}),"\n",(0,s.jsx)(t.h2,{id:"tldr",children:"TL;DR"}),"\n",(0,s.jsxs)(t.ul,{children:["\n",(0,s.jsx)(t.li,{children:"AutoGen\xae offers detailed multi-agent observability with AgentOps."}),"\n",(0,s.jsx)(t.li,{children:"AgentOps offers the best experience for developers building with AutoGen in just two lines of code."}),"\n",(0,s.jsx)(t.li,{children:"Enterprises can now trust AutoGen in production with detailed monitoring and logging from AgentOps."}),"\n"]}),"\n",(0,s.jsxs)(t.p,{children:["AutoGen is excited to announce an integration with AgentOps, the industry leader in agent observability and compliance. Back in February, ",(0,s.jsx)(t.a,{href:"https://www.bloomberg.com/news/newsletters/2024-02-15/tech-companies-bet-the-world-is-ready-for-ai-agents",children:"Bloomberg declared 2024 the year of AI Agents"}),". And it's true! We've seen AI transform from simplistic chatbots to autonomously making decisions and completing tasks on a user's behalf."]}),"\n",(0,s.jsx)(t.p,{children:"However, as with most new technologies, companies and engineering teams can be slow to develop processes and best practices. One part of the agent workflow we're betting on is the importance of observability. Letting your agents run wild might work for a hobby project, but if you're building enterprise-grade agents for production, it's crucial to understand where your agents are succeeding and failing. Observability isn't just an option; it's a requirement."}),"\n",(0,s.jsx)(t.p,{children:"As agents evolve into even more powerful and complex tools, you should view them increasingly as tools designed to augment your team's capabilities. Agents will take on more prominent roles and responsibilities, take action, and provide immense value. However, this means you must monitor your agents the same way a good manager maintains visibility over their personnel. AgentOps offers developers observability for debugging and detecting failures. It provides the tools to monitor all the key metrics your agents use in one easy-to-read dashboard. Monitoring is more than just a \u201cnice to have\u201d; it's a critical component for any team looking to build and scale AI agents."}),"\n",(0,s.jsx)(t.h2,{id:"what-is-agent-observability",children:"What is Agent Observability?"}),"\n",(0,s.jsx)(t.p,{children:"Agent observability, in its most basic form, allows you to monitor, troubleshoot, and clarify the actions of your agent during its operation. The ability to observe every detail of your agent's activity, right down to a timestamp, enables you to trace its actions precisely, identify areas for improvement, and understand the reasons behind any failures \u2014 a key aspect of effective debugging. Beyond enhancing diagnostic precision, this level of observability is integral for your system's reliability. Think of it as the ability to identify and address issues before they spiral out of control. Observability isn't just about keeping things running smoothly and maximizing uptime; it's about strengthening your agent-based solutions."}),"\n",(0,s.jsx)("img",{src:"https://github.com/AgentOps-AI/agentops/blob/main/docs/images/external/autogen/flow.png?raw=true",alt:"AI agent observability",style:{maxWidth:"100%"}}),"\n",(0,s.jsx)(t.h2,{id:"why-agentops",children:"Why AgentOps?"}),"\n",(0,s.jsxs)(t.p,{children:["AutoGen has simplified the process of building agents, yet we recognized the need for an easy-to-use, native tool for observability. We've previously discussed AgentOps, and now we're excited to partner with AgentOps as our official agent observability tool. Integrating AgentOps with AutoGen simplifies your workflow and boosts your agents' performance through clear observability, ensuring they operate optimally. For more details, check out our ",(0,s.jsx)(t.a,{href:"https://ag2ai.github.io/ag2/docs/notebooks/agentchat_agentops/",children:"AgentOps documentation"}),"."]}),"\n",(0,s.jsx)("img",{src:"https://github.com/AgentOps-AI/agentops/blob/main/docs/images/external/autogen/session-replay.png?raw=true",alt:"Agent Session Replay",style:{maxWidth:"100%"}}),"\n",(0,s.jsx)(t.p,{children:"Enterprises and enthusiasts trust AutoGen as the leader in building agents. With our partnership with AgentOps, developers can now natively debug agents for efficiency and ensure compliance, providing a comprehensive audit trail for all of your agents' activities. AgentOps allows you to monitor LLM calls, costs, latency, agent failures, multi-agent interactions, tool usage, session-wide statistics, and more all from one dashboard."}),"\n",(0,s.jsx)(t.p,{children:"By combining the agent-building capabilities of AutoGen with the observability tools of AgentOps, we're providing our users with a comprehensive solution that enhances agent performance and reliability. This collaboration establishes that enterprises can confidently deploy AI agents in production environments, knowing they have the best tools to monitor, debug, and optimize their agents."}),"\n",(0,s.jsxs)(t.p,{children:["The best part is that it only takes two lines of code. All you need to do is set an ",(0,s.jsx)(t.code,{children:"AGENTOPS_API_KEY"})," in your environment (Get API key here: ",(0,s.jsx)(t.a,{href:"https://app.agentops.ai/account",children:"https://app.agentops.ai/account"}),") and call ",(0,s.jsx)(t.code,{children:"agentops.init()"}),":"]}),"\n",(0,s.jsx)(t.pre,{children:(0,s.jsx)(t.code,{children:'import os\nimport agentops\n\nagentops.init(os.environ["AGENTOPS_API_KEY"])\n'})}),"\n",(0,s.jsx)(t.h2,{id:"agentopss-features",children:"AgentOps's Features"}),"\n",(0,s.jsx)(t.p,{children:"AgentOps includes all the functionality you need to ensure your agents are suitable for real-world, scalable solutions."}),"\n",(0,s.jsx)("img",{src:"https://github.com/AgentOps-AI/agentops/blob/main/docs/images/external/autogen/dashboard.png?raw=true",alt:"AgentOps overview dashboard",style:{maxWidth:"100%"}}),"\n",(0,s.jsxs)(t.ul,{children:["\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Analytics Dashboard:"})," The AgentOps Analytics Dashboard allows you to configure and assign agents and automatically track what actions each agent is taking simultaneously. When used with AutoGen, AgentOps is automatically configured for multi-agent compatibility, allowing users to track multiple agents across runs easily. Instead of a terminal-level screen, AgentOps provides a superior user experience with its intuitive interface."]}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Tracking LLM Costs:"})," Cost tracking is natively set up within AgentOps and provides a rolling total. This allows developers to see and track their run costs and accurately predict future costs."]}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Recursive Thought Detection:"})," One of the most frustrating aspects of agents is when they get trapped and perform the same task repeatedly for hours on end. AgentOps can identify when agents fall into infinite loops, ensuring efficiency and preventing wasteful computation."]}),"\n"]}),"\n",(0,s.jsx)(t.p,{children:"AutoGen users also have access to the following features in AgentOps:"}),"\n",(0,s.jsxs)(t.ul,{children:["\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Replay Analytics:"})," Watch step-by-step agent execution graphs."]}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Custom Reporting:"})," Create custom analytics on agent performance."]}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Public Model Testing:"})," Test your agents against benchmarks and leaderboards."]}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Custom Tests:"})," Run your agents against domain-specific tests."]}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Compliance and Security:"})," Create audit logs and detect potential threats, such as profanity and leaks of Personally Identifiable Information."]}),"\n",(0,s.jsxs)(t.li,{children:[(0,s.jsx)(t.strong,{children:"Prompt Injection Detection:"})," Identify potential code injection and secret leaks."]}),"\n"]}),"\n",(0,s.jsx)(t.h2,{id:"conclusion",children:"Conclusion"}),"\n",(0,s.jsx)(t.p,{children:"By integrating AgentOps into AutoGen, we've given our users everything they need to make production-grade agents, improve them, and track their performance to ensure they're doing exactly what you need them to do. Without it, you're operating blindly, unable to tell where your agents are succeeding or failing. AgentOps provides the required observability tools needed to monitor, debug, and optimize your agents for enterprise-level performance. It offers everything developers need to scale their AI solutions, from cost tracking to recursive thought detection."}),"\n",(0,s.jsxs)(t.p,{children:["Did you find this note helpful? Would you like to share your thoughts, use cases, and findings? Please join our observability channel in the ",(0,s.jsx)(t.a,{href:"https://discord.gg/hXJknP54EH",children:"AutoGen Discord"}),"."]})]})}function d(e={}){const{wrapper:t}={...(0,i.a)(),...e.components};return t?(0,s.jsx)(t,{...e,children:(0,s.jsx)(c,{...e})}):c(e)}},11151:(e,t,n)=>{n.d(t,{Z:()=>r,a:()=>a});var s=n(67294);const i={},o=s.createContext(i);function a(e){const t=s.useContext(o);return s.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function r(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:a(e.components),s.createElement(o.Provider,{value:t},e.children)}}}]);