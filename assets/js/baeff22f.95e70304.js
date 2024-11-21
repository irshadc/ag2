"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[34400],{13012:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>l,contentTitle:()=>o,default:()=>u,frontMatter:()=>s,metadata:()=>r,toc:()=>d});var n=a(85893),i=a(11151);const s={title:"Unlocking the Power of Agentic Workflows at Nexla with Autogen",authors:["Noel1997","ameydesai"],tags:["data automation","agents","Autogen","Nexla"]},o="Unlocking the Power of Agentic Workflows at Nexla with Autogen",r={permalink:"/ag2/blog/2024/10/23/NOVA",source:"@site/blog/2024-10-23-NOVA/index.mdx",title:"Unlocking the Power of Agentic Workflows at Nexla with Autogen",description:"nexlaautogen",date:"2024-10-23T00:00:00.000Z",formattedDate:"October 23, 2024",tags:[{label:"data automation",permalink:"/ag2/blog/tags/data-automation"},{label:"agents",permalink:"/ag2/blog/tags/agents"},{label:"Autogen",permalink:"/ag2/blog/tags/autogen"},{label:"Nexla",permalink:"/ag2/blog/tags/nexla"}],readingTime:5.55,hasTruncateMarker:!1,authors:[{name:"Noel Nebu Panicker",title:"AI Engineer at Nexla",url:"https://github.com/Noel1997",imageURL:"https://github.com/Noel1997.png",key:"Noel1997"},{name:"Amey Desai",title:"Head of AI at Nexla",url:"https://github.com/ameyitis",imageURL:"https://github.com/ameyitis.png",key:"ameydesai"}],frontMatter:{title:"Unlocking the Power of Agentic Workflows at Nexla with Autogen",authors:["Noel1997","ameydesai"],tags:["data automation","agents","Autogen","Nexla"]},unlisted:!1,prevItem:{title:"Introducing CaptainAgent for Adaptive Team Building",permalink:"/ag2/blog/2024/11/15/CaptainAgent"},nextItem:{title:"AgentOps, the Best Tool for AutoGen Agent Observability",permalink:"/ag2/blog/2024/07/25/AgentOps"}},l={authorsImageUrls:[void 0,void 0]},d=[{value:"The Challenge: Elevating Data Automation at Nexla",id:"the-challenge-elevating-data-automation-at-nexla",level:2},{value:"The Solution: Harnessing Autogen for Project NOVA",id:"the-solution-harnessing-autogen-for-project-nova",level:2},{value:"Natural Language to Transforms",id:"natural-language-to-transforms",level:2},{value:"Natural Language to ELT (Extract, Load and Transform)",id:"natural-language-to-elt-extract-load-and-transform",level:2},{value:"Use Cases: Empowering Diverse Users",id:"use-cases-empowering-diverse-users",level:2},{value:"Why Nexla and Autogen?",id:"why-nexla-and-autogen",level:2},{value:"Technical Deep Dive: The Architecture Behind NOVA",id:"technical-deep-dive-the-architecture-behind-nova",level:2},{value:"NOVA Architecture Diagram",id:"nova-architecture-diagram",level:2},{value:"Using Server-Sent Events (SSE) in NOVA",id:"using-server-sent-events-sse-in-nova",level:2},{value:"Conclusion: The Future of AI at Nexla",id:"conclusion-the-future-of-ai-at-nexla",level:2}];function c(e){const t={a:"a",h2:"h2",img:"img",li:"li",p:"p",strong:"strong",ul:"ul",...(0,i.a)(),...e.components};return(0,n.jsxs)(n.Fragment,{children:[(0,n.jsx)(t.p,{children:(0,n.jsx)(t.img,{alt:"nexla_autogen",src:a(29364).Z+"",width:"913",height:"480"})}),"\n",(0,n.jsxs)(t.p,{children:["In today\u2019s fast-paced GenAI landscape, organizations are constantly searching for smarter, more efficient ways to manage and transform data. ",(0,n.jsx)(t.a,{href:"https://nexla.com/",children:"Nexla"})," is a platform dedicated to the automation of data engineering, enabling users to get ready-to-use data with minimal hassle. Central to Nexla\u2019s approach are ",(0,n.jsx)(t.a,{href:"https://nexla.com/nexsets-modern-data-building-blocks/",children:"Nexsets"}),"\u2014data products that streamline the process of integrating, transforming, delivering, and monitoring data. Our mission is to make data ready-to-use for everyone, eliminating the complexities traditionally associated with data workflows."]}),"\n",(0,n.jsx)(t.p,{children:"With the introduction of Project NOVA, we\u2019re leveraging Autogen, an open-source project initialized by Microsoft and multiple academia institutions, to create powerful, production-grade agentic workflows that empower users to accomplish complex tasks with the simplicity of natural language."}),"\n",(0,n.jsx)(t.h2,{id:"the-challenge-elevating-data-automation-at-nexla",children:"The Challenge: Elevating Data Automation at Nexla"}),"\n",(0,n.jsx)(t.p,{children:"One of the primary challenges our customers face is the time and effort required to develop and manage complex data transformations. Even with a clear vision of the final data model, data transformation is a multi-step process that can be both time-consuming and technically demanding."}),"\n",(0,n.jsx)(t.h2,{id:"the-solution-harnessing-autogen-for-project-nova",children:"The Solution: Harnessing Autogen for Project NOVA"}),"\n",(0,n.jsx)(t.p,{children:"Autogen provided us with the perfect foundation to build intelligent agents capable of handling complex data tasks far beyond basic conversational functions. This led to the creation of NOVA\u2014Nexla Orchestrated Versatile Agents, a system designed to translate natural language into precise data transformations. NOVA simplifies data operations by breaking down complex tasks into manageable steps, enabling users to interact with their data intuitively and efficiently."}),"\n",(0,n.jsx)(t.p,{children:"By leveraging GenAI Agents built with Autogen, we\u2019ve also tackled the challenge of creating a common data model, allowing for seamless integration of diverse data sources to a unified data model. This innovation bridges the gap between user intent and data manipulation, paving the way for a unified and accessible data infrastructure across platforms and industries."}),"\n",(0,n.jsx)(t.h2,{id:"natural-language-to-transforms",children:"Natural Language to Transforms"}),"\n",(0,n.jsx)(t.p,{children:"NOVA\u2019s Natural Language to Transforms feature allows users to take a Nexset\u2014a data product within Nexla\u2014and describe, in plain language, the transformation they need. NOVA then automatically generates the required transforms, whether in Python or SQL, depending on the task."}),"\n",(0,n.jsx)(t.p,{children:'For example, a user could simply instruct, "Compute average speed and average duration for every origin-destination pair, hourly and by day of the week." NOVA breaks down this request into a series of steps, applies the necessary transformations, and delivers the desired output. This allows users to focus on analyzing and utilizing the transformed data without getting bogged down in the complexities of coding.'}),"\n",(0,n.jsx)(t.h2,{id:"natural-language-to-elt-extract-load-and-transform",children:"Natural Language to ELT (Extract, Load and Transform)"}),"\n",(0,n.jsx)(t.p,{children:"Next up is Natural Language to ELT, which allows users to build and execute ELT pipelines simply by providing natural language instructions. Users can input one or more Nexsets, a final data model, and an optional set of instructions, and NOVA does the rest."}),"\n",(0,n.jsx)(t.p,{children:"NOVA doesn\u2019t just generate a static script\u2014it allows users to interactively tweak the SQL logic as they go, ensuring that the final output is exactly what they need. This interactive, dynamic approach makes it easier than ever to handle complex ELT tasks, directly executing business logic on platforms like BigQuery or Snowflake and many other connectors that Nexla supports with precision and efficiency."}),"\n",(0,n.jsx)(t.h2,{id:"use-cases-empowering-diverse-users",children:"Use Cases: Empowering Diverse Users"}),"\n",(0,n.jsx)(t.p,{children:"These features are designed with a broad range of users in mind:"}),"\n",(0,n.jsxs)(t.ul,{children:["\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Data Engineers:"})," Automate routine data transformation tasks, freeing up time to focus on more strategic initiatives."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Business Analysts:"})," Generate insights quickly without the need for complex coding, enabling faster decision-making."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Business Users:"})," Interact with data naturally, transforming ideas into actionable queries without requiring deep technical expertise."]}),"\n"]}),"\n",(0,n.jsx)(t.h2,{id:"why-nexla-and-autogen",children:"Why Nexla and Autogen?"}),"\n",(0,n.jsx)(t.p,{children:"Nexla\u2019s unique value proposition is its ability to integrate advanced AI-driven automation into existing workflows seamlessly. By building on the robust capabilities of Autogen, we\u2019ve ensured that NOVA is not only scalable but also reliable for production-grade applications. The flexibility and power of Autogen have been instrumental in allowing us to create agents that handle sophisticated tasks beyond basic interactions, making them an essential part of our platform\u2019s evolution."}),"\n",(0,n.jsx)(t.p,{children:"Moreover, the scalability and reliability of Autogen have enabled us to deploy these features across large datasets and cloud platforms, ensuring consistent performance even under demanding workloads."}),"\n",(0,n.jsx)(t.h2,{id:"technical-deep-dive-the-architecture-behind-nova",children:"Technical Deep Dive: The Architecture Behind NOVA"}),"\n",(0,n.jsx)(t.p,{children:"At the heart of NOVA\u2019s success is a sophisticated agent architecture, powered by Autogen:"}),"\n",(0,n.jsxs)(t.ul,{children:["\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Planner Agent:"})," Analyzes user queries to determine the necessary steps for the ELT or transformation task, planning the workflow."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Query Interpreter Agent:"})," Translates the planner\u2019s high-level steps into actionable SQL or Python for execution by the Data Transformer Agent."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Data Transformer Agent:"})," Generates the required SQL or Python logic, ensuring it aligns with the specific schema and data samples."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"Evaluator Agent:"})," Reviews the generated logic for accuracy before execution, ensuring it meets the necessary requirements."]}),"\n",(0,n.jsxs)(t.li,{children:[(0,n.jsx)(t.strong,{children:"API Agent:"})," Manages interactions with databases and cloud services, executing the approved logic and creating Nexsets as needed."]}),"\n"]}),"\n",(0,n.jsx)(t.p,{children:"These agents work together to deliver a seamless, intuitive experience for users, automating tasks that would otherwise require significant manual effort and technical expertise."}),"\n",(0,n.jsx)(t.h2,{id:"nova-architecture-diagram",children:"NOVA Architecture Diagram"}),"\n",(0,n.jsx)(t.p,{children:(0,n.jsx)(t.img,{alt:"nova_architecture",src:a(43536).Z+"",width:"2182",height:"1832"})}),"\n",(0,n.jsx)(t.h2,{id:"using-server-sent-events-sse-in-nova",children:"Using Server-Sent Events (SSE) in NOVA"}),"\n",(0,n.jsx)(t.p,{children:"An essential component of NOVA's architecture is the use of Server-Sent Events (SSE) to maintain real-time communication between the backend agents and the user interface. As the agents work through the various stages of query analysis, transformation, and execution, SSE allows NOVA to stream live updates back to the user. This ensures that users receive timely feedback on the status of their requests, especially for complex, multi-step processes. By leveraging SSE, we enhance the overall user experience, making interactions with NOVA feel more dynamic and responsive, while also providing insights into the ongoing data operations."}),"\n",(0,n.jsx)(t.h2,{id:"conclusion-the-future-of-ai-at-nexla",children:"Conclusion: The Future of AI at Nexla"}),"\n",(0,n.jsx)(t.p,{children:"Our progress in developing NOVA has been significantly enhanced by utilizing the Autogen open-source library. Autogen\u2019s powerful capabilities have been instrumental in helping us create intelligent agents that transform how users interact with data. As Autogen and similar technologies continue to evolve, we\u2019re eager to explore new possibilities and innovations in the field of data automation."}),"\n",(0,n.jsx)(t.p,{children:"Project NOVA and its features\u2014Natural Language to Transforms and Natural Language to ELT\u2014are just the beginning of what we believe is possible with Autogen. We\u2019re already exploring new ways to expand these capabilities, making them even more powerful and user-friendly."}),"\n",(0,n.jsxs)(t.p,{children:["We invite you to explore these features and see firsthand how they can transform your workflows. Whether you\u2019re a developer, analyst, or business leader, the possibilities are vast with Nexla. You can start your ",(0,n.jsx)(t.a,{href:"https://nexla.com/free-trial/",children:"free trial"})," to see how our solutions can work for you."]}),"\n",(0,n.jsxs)(t.p,{children:["For any inquiries or further information, feel free to ",(0,n.jsx)(t.a,{href:"mailto:ai@nexla.com",children:"contact us"}),"."]})]})}function u(e={}){const{wrapper:t}={...(0,i.a)(),...e.components};return t?(0,n.jsx)(t,{...e,children:(0,n.jsx)(c,{...e})}):c(e)}},29364:(e,t,a)=>{a.d(t,{Z:()=>n});const n=a.p+"assets/images/nexla_autogen-e105b0dd9a1db16a51a10dc967a17357.png"},43536:(e,t,a)=>{a.d(t,{Z:()=>n});const n=a.p+"assets/images/nova_architecture-df5963075d6c54e1e3516ee6b81d33f8.png"},11151:(e,t,a)=>{a.d(t,{Z:()=>r,a:()=>o});var n=a(67294);const i={},s=n.createContext(i);function o(e){const t=n.useContext(s);return n.useMemo((function(){return"function"==typeof e?e(t):{...t,...e}}),[t,e])}function r(e){let t;return t=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:o(e.components),n.createElement(s.Provider,{value:t},e.children)}}}]);