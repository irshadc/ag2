"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[60793],{91734:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>c,contentTitle:()=>a,default:()=>d,frontMatter:()=>o,metadata:()=>r,toc:()=>l});var s=t(85893),i=t(11151);const o={custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_websockets.ipynb",description:"Websockets facilitate real-time, bidirectional communication between web clients and servers, enhancing the responsiveness and interactivity of AutoGen-powered applications.",source_notebook:"/notebook/agentchat_websockets.ipynb",tags:["websockets","streaming"],title:"Websockets: Streaming input and output using websockets"},a="Websockets: Streaming input and output using websockets",r={id:"notebooks/agentchat_websockets",title:"Websockets: Streaming input and output using websockets",description:"Websockets facilitate real-time, bidirectional communication between web clients and servers, enhancing the responsiveness and interactivity of AutoGen-powered applications.",source:"@site/docs/notebooks/agentchat_websockets.mdx",sourceDirName:"notebooks",slug:"/notebooks/agentchat_websockets",permalink:"/ag2/docs/notebooks/agentchat_websockets",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_websockets.ipynb",tags:[{label:"websockets",permalink:"/ag2/docs/tags/websockets"},{label:"streaming",permalink:"/ag2/docs/tags/streaming"}],version:"current",frontMatter:{custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_websockets.ipynb",description:"Websockets facilitate real-time, bidirectional communication between web clients and servers, enhancing the responsiveness and interactivity of AutoGen-powered applications.",source_notebook:"/notebook/agentchat_websockets.ipynb",tags:["websockets","streaming"],title:"Websockets: Streaming input and output using websockets"},sidebar:"notebooksSidebar",previous:{title:"Web Scraping using Apify Tools",permalink:"/ag2/docs/notebooks/agentchat_webscraping_with_apify"},next:{title:"Agent with memory using Mem0",permalink:"/ag2/docs/notebooks/agentchat_with_memory"}},c={},l=[{value:"Requirements",id:"requirements",level:2},{value:"Set your API Endpoint",id:"set-your-api-endpoint",level:2},{value:"Defining <code>on_connect</code> function",id:"defining-on_connect-function",level:2},{value:"Testing websockets server with Python client",id:"testing-websockets-server-with-python-client",level:2},{value:"Testing websockets server running inside FastAPI server with HTML/JS client",id:"testing-websockets-server-running-inside-fastapi-server-with-htmljs-client",level:2},{value:"Testing websockets server with HTML/JS client",id:"testing-websockets-server-with-htmljs-client",level:2}];function h(e){const n={a:"a",admonition:"admonition",code:"code",h1:"h1",h2:"h2",img:"img",li:"li",ol:"ol",p:"p",pre:"pre",strong:"strong",...(0,i.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.h1,{id:"websockets-streaming-input-and-output-using-websockets",children:"Websockets: Streaming input and output using websockets"}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.a,{href:"https://colab.research.google.com/github/ag2ai/ag2/blob/main/notebook/agentchat_websockets.ipynb",children:(0,s.jsx)(n.img,{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Open In Colab"})}),"\n",(0,s.jsx)(n.a,{href:"https://github.com/ag2ai/ag2/blob/main/notebook/agentchat_websockets.ipynb",children:(0,s.jsx)(n.img,{src:"https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github",alt:"Open on GitHub"})})]}),"\n",(0,s.jsxs)(n.p,{children:["This notebook demonstrates how to use the\n",(0,s.jsx)(n.a,{href:"https://ag2ai.github.io/ag2/docs/reference/io/base/IOStream",children:(0,s.jsx)(n.code,{children:"IOStream"})}),"\nclass to stream both input and output using websockets. The use of\nwebsockets allows you to build web clients that are more responsive than\nthe one using web methods. The main difference is that the webosockets\nallows you to push data while you need to poll the server for new\nresponse using web mothods."]}),"\n",(0,s.jsxs)(n.p,{children:["In this guide, we explore the capabilities of the\n",(0,s.jsx)(n.a,{href:"https://ag2ai.github.io/ag2/docs/reference/io/base/IOStream",children:(0,s.jsx)(n.code,{children:"IOStream"})}),"\nclass. It is specifically designed to enhance the development of clients\nsuch as web clients which use websockets for streaming both input and\noutput. The\n",(0,s.jsx)(n.a,{href:"https://ag2ai.github.io/ag2/docs/reference/io/base/IOStream",children:(0,s.jsx)(n.code,{children:"IOStream"})}),"\nstands out by enabling a more dynamic and interactive user experience\nfor web applications."]}),"\n",(0,s.jsxs)(n.p,{children:["Websockets technology is at the core of this functionality, offering a\nsignificant advancement over traditional web methods by allowing data to\nbe \u201cpushed\u201d to the client in real-time. This is a departure from the\nconventional approach where clients must repeatedly \u201cpoll\u201d the server to\ncheck for any new responses. By employing the underlining\n",(0,s.jsx)(n.a,{href:"https://websockets.readthedocs.io/",children:"websockets"})," library, the IOStream\nclass facilitates a continuous, two-way communication channel between\nthe server and client. This ensures that updates are received instantly,\nwithout the need for constant polling, thereby making web clients more\nefficient and responsive."]}),"\n",(0,s.jsxs)(n.p,{children:["The real power of websockets, leveraged through the\n",(0,s.jsx)(n.a,{href:"https://ag2ai.github.io/ag2/docs/reference/io/base/IOStream",children:(0,s.jsx)(n.code,{children:"IOStream"})}),"\nclass, lies in its ability to create highly responsive web clients. This\nresponsiveness is critical for applications requiring real-time data\nupdates such as chat applications. By integrating the\n",(0,s.jsx)(n.a,{href:"https://ag2ai.github.io/ag2/docs/reference/io/base/IOStream",children:(0,s.jsx)(n.code,{children:"IOStream"})}),"\nclass into your web application, you not only enhance user experience\nthrough immediate data transmission but also reduce the load on your\nserver by eliminating unnecessary polling."]}),"\n",(0,s.jsxs)(n.p,{children:["In essence, the transition to using websockets through the\n",(0,s.jsx)(n.a,{href:"https://ag2ai.github.io/ag2/docs/reference/io/base/IOStream",children:(0,s.jsx)(n.code,{children:"IOStream"})}),"\nclass marks a significant enhancement in web client development. This\napproach not only streamlines the data exchange process between clients\nand servers but also opens up new possibilities for creating more\ninteractive and engaging web applications. By following this guide,\ndevelopers can harness the full potential of websockets and the\n",(0,s.jsx)(n.a,{href:"https://ag2ai.github.io/ag2/docs/reference/io/base/IOStream",children:(0,s.jsx)(n.code,{children:"IOStream"})}),"\nclass to push the boundaries of what is possible with web client\nresponsiveness and interactivity."]}),"\n",(0,s.jsx)(n.h2,{id:"requirements",children:"Requirements"}),"\n",(0,s.jsxs)(n.admonition,{title:"Requirements",type:"info",children:[(0,s.jsx)(n.p,{children:"Some extra dependencies are needed for this notebook, which can be installed via pip:"}),(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-bash",children:"pip install autogen[websockets] fastapi uvicorn\n"})}),(0,s.jsxs)(n.p,{children:["For more information, please refer to the ",(0,s.jsx)(n.a,{href:"/docs/installation/",children:"installation guide"}),"."]})]}),"\n",(0,s.jsx)(n.h2,{id:"set-your-api-endpoint",children:"Set your API Endpoint"}),"\n",(0,s.jsxs)(n.p,{children:["The\n",(0,s.jsx)(n.a,{href:"https://ag2ai.github.io/ag2/docs/reference/oai/openai_utils#config_list_from_json",children:(0,s.jsx)(n.code,{children:"config_list_from_json"})}),"\nfunction loads a list of configurations from an environment variable or\na json file."]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'from datetime import datetime\nfrom tempfile import TemporaryDirectory\n\nfrom websockets.sync.client import connect as ws_connect\n\nimport autogen\nfrom autogen.io.websockets import IOWebsockets\n\nconfig_list = autogen.config_list_from_json(\n    env_or_file="OAI_CONFIG_LIST",\n    filter_dict={\n        "model": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],\n    },\n)\n'})}),"\n",(0,s.jsx)(n.admonition,{type:"tip",children:(0,s.jsxs)(n.p,{children:["Learn more about configuring LLMs for agents ",(0,s.jsx)(n.a,{href:"/docs/topics/llm_configuration",children:"here"}),"."]})}),"\n",(0,s.jsxs)(n.h2,{id:"defining-on_connect-function",children:["Defining ",(0,s.jsx)(n.code,{children:"on_connect"})," function"]}),"\n",(0,s.jsxs)(n.p,{children:["An ",(0,s.jsx)(n.code,{children:"on_connect"})," function is a crucial part of applications that utilize\nwebsockets, acting as an event handler that is called whenever a new\nclient connection is established. This function is designed to initiate\nany necessary setup, communication protocols, or data exchange\nprocedures specific to the newly connected client. Essentially, it lays\nthe groundwork for the interactive session that follows, configuring how\nthe server and the client will communicate and what initial actions are\nto be taken once a connection is made. Now, let\u2019s delve into the details\nof how to define this function, especially in the context of using the\nAutoGen framework with websockets."]}),"\n",(0,s.jsxs)(n.p,{children:["Upon a client\u2019s connection to the websocket server, the server\nautomatically initiates a new instance of the\n",(0,s.jsx)(n.a,{href:"https://ag2ai.github.io/ag2/docs/reference/io/websockets/IOWebsockets",children:(0,s.jsx)(n.code,{children:"IOWebsockets"})}),"\nclass. This instance is crucial for managing the data flow between the\nserver and the client. The ",(0,s.jsx)(n.code,{children:"on_connect"})," function leverages this instance\nto set up the communication protocol, define interaction rules, and\ninitiate any preliminary data exchanges or configurations required for\nthe client-server interaction to proceed smoothly."]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'def on_connect(iostream: IOWebsockets) -> None:\n    print(f" - on_connect(): Connected to client using IOWebsockets {iostream}", flush=True)\n\n    print(" - on_connect(): Receiving message from client.", flush=True)\n\n    # 1. Receive Initial Message\n    initial_msg = iostream.input()\n\n    # 2. Instantiate ConversableAgent\n    agent = autogen.ConversableAgent(\n        name="chatbot",\n        system_message="Complete a task given to you and reply TERMINATE when the task is done. If asked about the weather, use tool \'weather_forecast(city)\' to get the weather forecast for a city.",\n        llm_config={\n            "config_list": autogen.config_list_from_json(\n                env_or_file="OAI_CONFIG_LIST",\n                filter_dict={\n                    "model": ["gpt-4", "gpt-3.5-turbo", "gpt-3.5-turbo-16k"],\n                },\n            ),\n            "stream": True,\n        },\n    )\n\n    # 3. Define UserProxyAgent\n    user_proxy = autogen.UserProxyAgent(\n        name="user_proxy",\n        system_message="A proxy for the user.",\n        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),\n        human_input_mode="NEVER",\n        max_consecutive_auto_reply=10,\n        code_execution_config=False,\n    )\n\n    # 4. Define Agent-specific Functions\n    def weather_forecast(city: str) -> str:\n        return f"The weather forecast for {city} at {datetime.now()} is sunny."\n\n    autogen.register_function(\n        weather_forecast, caller=agent, executor=user_proxy, description="Weather forecast for a city"\n    )\n\n    # 5. Initiate conversation\n    print(\n        f" - on_connect(): Initiating chat with agent {agent} using message \'{initial_msg}\'",\n        flush=True,\n    )\n    user_proxy.initiate_chat(  # noqa: F704\n        agent,\n        message=initial_msg,\n    )\n'})}),"\n",(0,s.jsxs)(n.p,{children:["Here\u2019s an explanation on how a typical ",(0,s.jsx)(n.code,{children:"on_connect"})," function such as the\none in the example above is defined:"]}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Receive Initial Message"}),": Immediately after establishing a\nconnection, receive an initial message from the client. This step is\ncrucial for understanding the client\u2019s request or initiating the\nconversation flow."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Instantiate ConversableAgent"}),": Create an instance of\nConversableAgent with a specific system message and the LLM\nconfiguration. If you need more than one agent, make sure they don\u2019t\nshare the same ",(0,s.jsx)(n.code,{children:"llm_config"})," as adding a function to one of them will\nalso attempt to add it to another."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Instantiate UserProxyAgent"}),": Similarly, create a UserProxyAgent\ninstance, defining its termination condition, human input mode, and\nother relevant parameters. There is no need to define ",(0,s.jsx)(n.code,{children:"llm_config"}),"\nas the UserProxyAgent does not use LLM."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Define Agent-specific Functions"}),": If your conversable agent\nrequires executing specific tasks, such as fetching a weather\nforecast in the example above, define these functions within the\non_connect scope. Decorate these functions accordingly to link them\nwith your agents."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Initiate Conversation"}),": Finally, use the ",(0,s.jsx)(n.code,{children:"initiate_chat"})," method\nof your ",(0,s.jsx)(n.code,{children:"UserProxyAgent"})," to start the interaction with the\nconversable agent, passing the initial message and a cache mechanism\nfor efficiency."]}),"\n"]}),"\n"]}),"\n",(0,s.jsx)(n.h2,{id:"testing-websockets-server-with-python-client",children:"Testing websockets server with Python client"}),"\n",(0,s.jsxs)(n.p,{children:["Testing an ",(0,s.jsx)(n.code,{children:"on_connect"})," function with a Python client involves\nsimulating a client-server interaction to ensure the setup, data\nexchange, and communication protocols function as intended. Here\u2019s a\nbrief explanation on how to conduct this test using a Python client:"]}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Start the Websocket Server"}),": Use the\n",(0,s.jsx)(n.code,{children:"IOWebsockets.run_server_in_thread method"})," to start the server in a\nseparate thread, specifying the on_connect function and the port.\nThis method returns the URI of the running websocket server."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Connect to the Server"}),": Open a connection to the server using the\nreturned URI. This simulates a client initiating a connection to\nyour websocket server."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Send a Message to the Server"}),": Once connected, send a message\nfrom the client to the server. This tests the server\u2019s ability to\nreceive messages through the established websocket connection."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Receive and Process Messages"}),": Implement a loop to continuously\nreceive messages from the server. Decode the messages if necessary,\nand process them accordingly. This step verifies the server\u2019s\nability to respond back to the client\u2019s request."]}),"\n"]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:["This test scenario effectively evaluates the interaction between a\nclient and a server using the ",(0,s.jsx)(n.code,{children:"on_connect"})," function, by simulating a\nrealistic message exchange. It ensures that the server can handle\nincoming connections, process messages, and communicate responses back\nto the client, all critical functionalities for a robust websocket-based\napplication."]}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=8765) as uri:\n    print(f" - test_setup() with websocket server running on {uri}.", flush=True)\n\n    with ws_connect(uri) as websocket:\n        print(f" - Connected to server on {uri}", flush=True)\n\n        print(" - Sending message to server.", flush=True)\n        # websocket.send("2+2=?")\n        websocket.send("Check out the weather in Paris and write a poem about it.")\n\n        while True:\n            message = websocket.recv()\n            message = message.decode("utf-8") if isinstance(message, bytes) else message\n\n            print(message, end="", flush=True)\n\n            if "TERMINATE" in message:\n                print()\n                print(" - Received TERMINATE message. Exiting.", flush=True)\n                break\n'})}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-text",children:" - test_setup() with websocket server running on ws://127.0.0.1:8765.\n - on_connect(): Connected to client using IOWebsockets <autogen.io.websockets.IOWebsockets object at 0x7b8fd65b3c10>\n - on_connect(): Receiving message from client.\n - Connected to server on ws://127.0.0.1:8765\n - Sending message to server.\n - on_connect(): Initiating chat with agent <autogen.agentchat.conversable_agent.ConversableAgent object at 0x7b909c086290> using message 'Check out the weather in Paris and write a poem about it.'\nuser_proxy (to chatbot):\n\nCheck out the weather in Paris and write a poem about it.\n\n--------------------------------------------------------------------------------\n\n>>>>>>>> USING AUTO REPLY...\nchatbot (to user_proxy):\n\n\n***** Suggested tool call (call_xFFWe52vwdpgZ8xTRV6adBdy): weather_forecast *****\nArguments: \n{\n  \"city\": \"Paris\"\n}\n*********************************************************************************\n\n--------------------------------------------------------------------------------\n\n>>>>>>>> EXECUTING FUNCTION weather_forecast...\nuser_proxy (to chatbot):\n\nuser_proxy (to chatbot):\n\n***** Response from calling tool (call_xFFWe52vwdpgZ8xTRV6adBdy) *****\nThe weather forecast for Paris at 2024-04-05 12:00:06.206125 is sunny.\n**********************************************************************\n\n--------------------------------------------------------------------------------\n\n>>>>>>>> USING AUTO REPLY...\nIn the heart of France, beneath the sun's warm glow,\nLies the city of Paris, where the Seine waters flow.\nBathed in sunlight, every street and spire,\nIlluminated each detail, just like a docile fire.\n\nOnce monochromatic cityscape, kissed by the sun's bright light,\nNow a kaleidoscope of colors, from morning till the night.\nThis sun-swept city sparkles, under the azure dome,\nHer inhabitants find comfort, for they call this city home.\n\nOne can wander in her sunshine, on this perfect weather day,\nAnd feel the warmth it brings, to chase your blues away.\nFor the weather in Paris, is more than just a forecast,\nIt is a stage setting for dwellers and tourists amassed.\n\nTERMINATE\n\nchatbot (to user_proxy):\n\nIn the heart of France, beneath the sun's warm glow,\nLies the city of Paris, where the Seine waters flow.\nBathed in sunlight, every street and spire,\nIlluminated each detail, just like a docile fire.\n\nOnce monochromatic cityscape, kissed by the sun's bright light,\nNow a kaleidoscope of colors, from morning till the night.\nThis sun-swept city sparkles, under the azure dome,\nHer inhabitants find comfort, for they call this city home.\n\nOne can wander in her sunshine, on this perfect weather day,\nAnd feel the warmth it brings, to chase your blues away.\nFor the weather in Paris, is more than just a forecast,\nIt is a stage setting for dwellers and tourists amassed.\n\nTERMINATE\n\n - Received TERMINATE message. Exiting.\n"})}),"\n",(0,s.jsx)(n.h2,{id:"testing-websockets-server-running-inside-fastapi-server-with-htmljs-client",children:"Testing websockets server running inside FastAPI server with HTML/JS client"}),"\n",(0,s.jsxs)(n.p,{children:["The code snippets below outlines an approach for testing an ",(0,s.jsx)(n.code,{children:"on_connect"}),"\nfunction in a web environment using\n",(0,s.jsx)(n.a,{href:"https://fastapi.tiangolo.com/",children:"FastAPI"})," to serve a simple interactive\nHTML page. This method allows users to send messages through a web\ninterface, which are then processed by the server running the AutoGen\nframework via websockets. Here\u2019s a step-by-step explanation:"]}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"FastAPI Application Setup"}),": The code initiates by importing\nnecessary libraries and setting up a FastAPI application. FastAPI is\na modern, fast web framework for building APIs with Python 3.7+\nbased on standard Python type hints."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"HTML Template for User Interaction"}),": An HTML template is defined\nas a multi-line Python string, which includes a basic form for\nmessage input and a script for managing websocket communication.\nThis template creates a user interface where messages can be sent to\nthe server and responses are displayed dynamically."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Running the Websocket Server"}),": The ",(0,s.jsx)(n.code,{children:"run_websocket_server"})," async\ncontext manager starts the websocket server using\n",(0,s.jsx)(n.code,{children:"IOWebsockets.run_server_in_thread"})," with the specified ",(0,s.jsx)(n.code,{children:"on_connect"}),"\nfunction and port. This server listens for incoming websocket\nconnections."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"FastAPI Route for Serving HTML Page"}),": A FastAPI route\n(",(0,s.jsx)(n.code,{children:'@app.get("/")'}),") is defined to serve the HTML page to users. When a\nuser accesses the root URL, the HTML content for the websocket chat\nis returned, allowing them to interact with the websocket server."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Starting the FastAPI Application"}),": Lastly, the FastAPI\napplication is started using Uvicorn, an ASGI server, configured\nwith the app and additional parameters as needed. The server is then\nlaunched to serve the FastAPI application, making the interactive\nHTML page accessible to users."]}),"\n"]}),"\n"]}),"\n",(0,s.jsx)(n.p,{children:"This method of testing allows for interactive communication between the\nuser and the server, providing a practical way to demonstrate and\nevaluate the behavior of the on_connect function in real-time. Users can\nsend messages through the webpage, and the server processes these\nmessages as per the logic defined in the on_connect function, showcasing\nthe capabilities and responsiveness of the AutoGen framework\u2019s websocket\nhandling in a user-friendly manner."}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'from contextlib import asynccontextmanager  # noqa: E402\nfrom pathlib import Path  # noqa: E402\n\nfrom fastapi import FastAPI  # noqa: E402\nfrom fastapi.responses import HTMLResponse  # noqa: E402\n\nPORT = 8000\n\nhtml = """\n<!DOCTYPE html>\n<html>\n    <head>\n        <title>Autogen websocket test</title>\n    </head>\n    <body>\n        <h1>WebSocket Chat</h1>\n        <form action="" onsubmit="sendMessage(event)">\n            <input type="text" id="messageText" autocomplete="off"/>\n            <button>Send</button>\n        </form>\n        <ul id=\'messages\'>\n        </ul>\n        <script>\n            var ws = new WebSocket("ws://localhost:8080/ws");\n            ws.onmessage = function(event) {\n                var messages = document.getElementById(\'messages\')\n                var message = document.createElement(\'li\')\n                var content = document.createTextNode(event.data)\n                message.appendChild(content)\n                messages.appendChild(message)\n            };\n            function sendMessage(event) {\n                var input = document.getElementById("messageText")\n                ws.send(input.value)\n                input.value = \'\'\n                event.preventDefault()\n            }\n        <\/script>\n    </body>\n</html>\n"""\n\n\n@asynccontextmanager\nasync def run_websocket_server(app):\n    with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=8080) as uri:\n        print(f"Websocket server started at {uri}.", flush=True)\n\n        yield\n\n\napp = FastAPI(lifespan=run_websocket_server)\n\n\n@app.get("/")\nasync def get():\n    return HTMLResponse(html)\n'})}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"import uvicorn  # noqa: E402\n\nconfig = uvicorn.Config(app)\nserver = uvicorn.Server(config)\nawait server.serve()  # noqa: F704\n"})}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-text",children:"INFO:     Started server process [5227]\nINFO:     Waiting for application startup.\nINFO:     Application startup complete.\nINFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)\nINFO:     Shutting down\nINFO:     Waiting for application shutdown.\nINFO:     Application shutdown complete.\nINFO:     Finished server process [5227]\n"})}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-text",children:'Websocket server started at ws://127.0.0.1:8080.\nINFO:     127.0.0.1:42548 - "GET / HTTP/1.1" 200 OK\nINFO:     127.0.0.1:42548 - "GET /favicon.ico HTTP/1.1" 404 Not Found\n - on_connect(): Connected to client using IOWebsockets <autogen.io.websockets.IOWebsockets object at 0x7b8fc6991420>\n - on_connect(): Receiving message from client.\n - on_connect(): Initiating chat with agent <autogen.agentchat.conversable_agent.ConversableAgent object at 0x7b909c0cab00> using message \'write a poem about lundon\'\n'})}),"\n",(0,s.jsx)(n.p,{children:"The testing setup described above, leveraging FastAPI and websockets,\nnot only serves as a robust testing framework for the on_connect\nfunction but also lays the groundwork for developing real-world\napplications. This approach exemplifies how web-based interactions can\nbe made dynamic and real-time, a critical aspect of modern application\ndevelopment."}),"\n",(0,s.jsx)(n.p,{children:"For instance, this setup can be directly applied or adapted to build\ninteractive chat applications, real-time data dashboards, or live\nsupport systems. The integration of websockets enables the server to\npush updates to clients instantly, a key feature for applications that\nrely on the timely delivery of information. For example, a chat\napplication built on this framework can support instantaneous messaging\nbetween users, enhancing user engagement and satisfaction."}),"\n",(0,s.jsx)(n.p,{children:"Moreover, the simplicity and interactivity of the HTML page used for\ntesting reflect how user interfaces can be designed to provide seamless\nexperiences. Developers can expand upon this foundation to incorporate\nmore sophisticated elements such as user authentication, message\nencryption, and custom user interactions, further tailoring the\napplication to meet specific use case requirements."}),"\n",(0,s.jsx)(n.p,{children:"The flexibility of the FastAPI framework, combined with the real-time\ncommunication enabled by websockets, provides a powerful toolset for\ndevelopers looking to build scalable, efficient, and highly interactive\nweb applications. Whether it\u2019s for creating collaborative platforms,\nstreaming services, or interactive gaming experiences, this testing\nsetup offers a glimpse into the potential applications that can be\ndeveloped with these technologies."}),"\n",(0,s.jsx)(n.h2,{id:"testing-websockets-server-with-htmljs-client",children:"Testing websockets server with HTML/JS client"}),"\n",(0,s.jsxs)(n.p,{children:["The provided code snippet below is an example of how to create an\ninteractive testing environment for an ",(0,s.jsx)(n.code,{children:"on_connect"})," function using\nPython\u2019s built-in ",(0,s.jsx)(n.code,{children:"http.server"})," module. This setup allows for real-time\ninteraction within a web browser, enabling developers to test the\nwebsocket functionality in a more user-friendly and practical manner.\nHere\u2019s a breakdown of how this code operates and its potential\napplications:"]}),"\n",(0,s.jsxs)(n.ol,{children:["\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Serving a Simple HTML Page"}),": The code starts by defining an HTML\npage that includes a form for sending messages and a list to display\nincoming messages. JavaScript is used to handle the form submission\nand websocket communication."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Temporary Directory for HTML File"}),": A temporary directory is\ncreated to store the HTML file. This approach ensures that the\ntesting environment is clean and isolated, minimizing conflicts with\nexisting files or configurations."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Custom HTTP Request Handler"}),": A custom subclass of\n",(0,s.jsx)(n.code,{children:"SimpleHTTPRequestHandler"})," is defined to serve the HTML file. This\nhandler overrides the do_GET method to redirect the root path (",(0,s.jsx)(n.code,{children:"/"}),")\nto the ",(0,s.jsx)(n.code,{children:"chat.html"})," page, ensuring that visitors to the server\u2019s root\nURL are immediately presented with the chat interface."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Starting the Websocket Server"}),": Concurrently, a websocket server\nis started on a different port using the\n",(0,s.jsx)(n.code,{children:"IOWebsockets.run_server_in_thread"})," method, with the previously\ndefined ",(0,s.jsx)(n.code,{children:"on_connect"})," function as the callback for new connections."]}),"\n"]}),"\n",(0,s.jsxs)(n.li,{children:["\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"HTTP Server for the HTML Interface"}),": An HTTP server is\ninstantiated to serve the HTML chat interface, enabling users to\ninteract with the websocket server through a web browser."]}),"\n"]}),"\n"]}),"\n",(0,s.jsx)(n.p,{children:"This setup showcases a practical application of integrating websockets\nwith a simple HTTP server to create a dynamic and interactive web\napplication. By using Python\u2019s standard library modules, it demonstrates\na low-barrier entry to developing real-time applications such as chat\nsystems, live notifications, or interactive dashboards."}),"\n",(0,s.jsx)(n.p,{children:"The key takeaway from this code example is how easily Python\u2019s built-in\nlibraries can be leveraged to prototype and test complex web\nfunctionalities. For developers looking to build real-world\napplications, this approach offers a straightforward method to validate\nand refine websocket communication logic before integrating it into\nlarger frameworks or systems. The simplicity and accessibility of this\ntesting setup make it an excellent starting point for developing a wide\nrange of interactive web applications."}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'from http.server import HTTPServer, SimpleHTTPRequestHandler  # noqa: E402\n\nPORT = 8000\n\nhtml = """\n<!DOCTYPE html>\n<html>\n    <head>\n        <title>Autogen websocket test</title>\n    </head>\n    <body>\n        <h1>WebSocket Chat</h1>\n        <form action="" onsubmit="sendMessage(event)">\n            <input type="text" id="messageText" autocomplete="off"/>\n            <button>Send</button>\n        </form>\n        <ul id=\'messages\'>\n        </ul>\n        <script>\n            var ws = new WebSocket("ws://localhost:8080/ws");\n            ws.onmessage = function(event) {\n                var messages = document.getElementById(\'messages\')\n                var message = document.createElement(\'li\')\n                var content = document.createTextNode(event.data)\n                message.appendChild(content)\n                messages.appendChild(message)\n            };\n            function sendMessage(event) {\n                var input = document.getElementById("messageText")\n                ws.send(input.value)\n                input.value = \'\'\n                event.preventDefault()\n            }\n        <\/script>\n    </body>\n</html>\n"""\n\nwith TemporaryDirectory() as temp_dir:\n    # create a simple HTTP webpage\n    path = Path(temp_dir) / "chat.html"\n    with open(path, "w") as f:\n        f.write(html)\n\n    #\n    class MyRequestHandler(SimpleHTTPRequestHandler):\n        def __init__(self, *args, **kwargs):\n            super().__init__(*args, directory=temp_dir, **kwargs)\n\n        def do_GET(self):\n            if self.path == "/":\n                self.path = "/chat.html"\n            return SimpleHTTPRequestHandler.do_GET(self)\n\n    handler = MyRequestHandler\n\n    with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=8080) as uri:\n        print(f"Websocket server started at {uri}.", flush=True)\n\n        with HTTPServer(("", PORT), handler) as httpd:\n            print("HTTP server started at http://localhost:" + str(PORT))\n            try:\n                httpd.serve_forever()\n            except KeyboardInterrupt:\n                print(" - HTTP server stopped.", flush=True)\n'})}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-text",children:"Websocket server started at ws://127.0.0.1:8080.\nHTTP server started at http://localhost:8000\n - on_connect(): Connected to client using IOWebsockets <autogen.io.websockets.IOWebsockets object at 0x7b8fc69937f0>\n - on_connect(): Receiving message from client.\n - on_connect(): Initiating chat with agent <autogen.agentchat.conversable_agent.ConversableAgent object at 0x7b8fc6990310> using message 'write a poem about new york'\n - on_connect(): Connected to client using IOWebsockets <autogen.io.websockets.IOWebsockets object at 0x7b8fc68bdc90>\n - on_connect(): Receiving message from client.\n - on_connect(): Initiating chat with agent <autogen.agentchat.conversable_agent.ConversableAgent object at 0x7b8fc68be170> using message 'check the weather in london and write a poem about it'\n - HTTP server stopped.\n"})}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-text",children:'127.0.0.1 - - [05/Apr/2024 12:01:51] "GET / HTTP/1.1" 200 -\n127.0.0.1 - - [05/Apr/2024 12:01:51] "GET / HTTP/1.1" 200 -\n127.0.0.1 - - [05/Apr/2024 12:02:27] "GET / HTTP/1.1" 304 -\n'})})]})}function d(e={}){const{wrapper:n}={...(0,i.a)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(h,{...e})}):h(e)}},11151:(e,n,t)=>{t.d(n,{Z:()=>r,a:()=>a});var s=t(67294);const i={},o=s.createContext(i);function a(e){const n=s.useContext(o);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:a(e.components),s.createElement(o.Provider,{value:n},e.children)}}}]);