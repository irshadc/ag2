"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[79960],{30790:(e,n,t)=>{t.r(n),t.d(n,{assets:()=>g,contentTitle:()=>s,default:()=>m,frontMatter:()=>o,metadata:()=>r,toc:()=>l});var a=t(85893),i=t(11151);const o={custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_image_generation_capability.ipynb",description:"Generate images with conversable agents.",source_notebook:"/notebook/agentchat_image_generation_capability.ipynb",tags:["capability","multimodal"],title:"Generate Dalle Images With Conversable Agents"},s="Generate Dalle Images With Conversable Agents",r={id:"notebooks/agentchat_image_generation_capability",title:"Generate Dalle Images With Conversable Agents",description:"Generate images with conversable agents.",source:"@site/docs/notebooks/agentchat_image_generation_capability.mdx",sourceDirName:"notebooks",slug:"/notebooks/agentchat_image_generation_capability",permalink:"/ag2/docs/notebooks/agentchat_image_generation_capability",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_image_generation_capability.ipynb",tags:[{label:"capability",permalink:"/ag2/docs/tags/capability"},{label:"multimodal",permalink:"/ag2/docs/tags/multimodal"}],version:"current",frontMatter:{custom_edit_url:"https://github.com/ag2ai/ag2/edit/main/notebook/agentchat_image_generation_capability.ipynb",description:"Generate images with conversable agents.",source_notebook:"/notebook/agentchat_image_generation_capability.ipynb",tags:["capability","multimodal"],title:"Generate Dalle Images With Conversable Agents"},sidebar:"notebooksSidebar",previous:{title:"Auto Generated Agent Chat: Task Solving with Code Generation, Execution, Debugging & Human Feedback",permalink:"/ag2/docs/notebooks/agentchat_human_feedback"},next:{title:"Auto Generated Agent Chat: Function Inception",permalink:"/ag2/docs/notebooks/agentchat_inception_function"}},g={},l=[];function c(e){const n={a:"a",admonition:"admonition",code:"code",h1:"h1",img:"img",p:"p",pre:"pre",...(0,i.a)(),...e.components};return(0,a.jsxs)(a.Fragment,{children:[(0,a.jsx)(n.h1,{id:"generate-dalle-images-with-conversable-agents",children:"Generate Dalle Images With Conversable Agents"}),"\n",(0,a.jsxs)(n.p,{children:[(0,a.jsx)(n.a,{href:"https://colab.research.google.com/github/ag2ai/ag2/blob/main/notebook/agentchat_image_generation_capability.ipynb",children:(0,a.jsx)(n.img,{src:"https://colab.research.google.com/assets/colab-badge.svg",alt:"Open In Colab"})}),"\n",(0,a.jsx)(n.a,{href:"https://github.com/ag2ai/ag2/blob/main/notebook/agentchat_image_generation_capability.ipynb",children:(0,a.jsx)(n.img,{src:"https://img.shields.io/badge/Open%20on%20GitHub-grey?logo=github",alt:"Open on GitHub"})})]}),"\n",(0,a.jsx)(n.p,{children:"This notebook illustrates how to add the image generation capability to\na conversable agent."}),"\n",(0,a.jsxs)(n.admonition,{title:"Requirements",type:"info",children:[(0,a.jsx)(n.p,{children:"Some extra dependencies are needed for this notebook, which can be installed via pip:"}),(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-bash",children:"pip install pyautogen[lmm]\n"})}),(0,a.jsxs)(n.p,{children:["For more information, please refer to the ",(0,a.jsx)(n.a,{href:"/docs/installation/",children:"installation guide"}),"."]})]}),"\n",(0,a.jsx)(n.p,{children:"First, let\u2019s import all the required modules to run this example."}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"import os\nimport re\nfrom typing import Dict, Optional\n\nfrom IPython.display import display\nfrom PIL.Image import Image\n\nimport autogen\nfrom autogen.agentchat.contrib import img_utils\nfrom autogen.agentchat.contrib.capabilities import generate_images\nfrom autogen.cache import Cache\nfrom autogen.oai import openai_utils\n"})}),"\n",(0,a.jsx)(n.p,{children:"Let\u2019s define our LLM configs."}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'gpt_config = {\n    "config_list": [{"model": "gpt-4-turbo-preview", "api_key": os.environ["OPENAI_API_KEY"]}],\n    "timeout": 120,\n    "temperature": 0.7,\n}\ngpt_vision_config = {\n    "config_list": [{"model": "gpt-4-vision-preview", "api_key": os.environ["OPENAI_API_KEY"]}],\n    "timeout": 120,\n    "temperature": 0.7,\n}\ndalle_config = {\n    "config_list": [{"model": "dall-e-3", "api_key": os.environ["OPENAI_API_KEY"]}],\n    "timeout": 120,\n    "temperature": 0.7,\n}\n'})}),"\n",(0,a.jsx)(n.admonition,{type:"tip",children:(0,a.jsxs)(n.p,{children:["Learn more about configuring LLMs for agents ",(0,a.jsx)(n.a,{href:"/docs/topics/llm_configuration",children:"here"}),"."]})}),"\n",(0,a.jsx)(n.p,{children:"Our system will consist of 2 main agents: 1. Image generator agent. 2.\nCritic agent."}),"\n",(0,a.jsx)(n.p,{children:"The image generator agent will carry a conversation with the critic, and\ngenerate images based on the critic\u2019s requests."}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'CRITIC_SYSTEM_MESSAGE = """You need to improve the prompt of the figures you saw.\nHow to create an image that is better in terms of color, shape, text (clarity), and other things.\nReply with the following format:\n\nCRITICS: the image needs to improve...\nPROMPT: here is the updated prompt!\n\nIf you have no critique or a prompt, just say TERMINATE\n"""\n'})}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'def _is_termination_message(msg) -> bool:\n    # Detects if we should terminate the conversation\n    if isinstance(msg.get("content"), str):\n        return msg["content"].rstrip().endswith("TERMINATE")\n    elif isinstance(msg.get("content"), list):\n        for content in msg["content"]:\n            if isinstance(content, dict) and "text" in content:\n                return content["text"].rstrip().endswith("TERMINATE")\n    return False\n\n\ndef critic_agent() -> autogen.ConversableAgent:\n    return autogen.ConversableAgent(\n        name="critic",\n        llm_config=gpt_vision_config,\n        system_message=CRITIC_SYSTEM_MESSAGE,\n        max_consecutive_auto_reply=3,\n        human_input_mode="NEVER",\n        is_termination_msg=lambda msg: _is_termination_message(msg),\n    )\n\n\ndef image_generator_agent() -> autogen.ConversableAgent:\n    # Create the agent\n    agent = autogen.ConversableAgent(\n        name="dalle",\n        llm_config=gpt_vision_config,\n        max_consecutive_auto_reply=3,\n        human_input_mode="NEVER",\n        is_termination_msg=lambda msg: _is_termination_message(msg),\n    )\n\n    # Add image generation ability to the agent\n    dalle_gen = generate_images.DalleImageGenerator(llm_config=dalle_config)\n    image_gen_capability = generate_images.ImageGeneration(\n        image_generator=dalle_gen, text_analyzer_llm_config=gpt_config\n    )\n\n    image_gen_capability.add_to_agent(agent)\n    return agent\n'})}),"\n",(0,a.jsxs)(n.p,{children:["We\u2019ll define ",(0,a.jsx)(n.code,{children:"extract_img"})," to help us extract the image generated by the\nimage generator agent."]}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'def extract_images(sender: autogen.ConversableAgent, recipient: autogen.ConversableAgent) -> Image:\n    images = []\n    all_messages = sender.chat_messages[recipient]\n\n    for message in reversed(all_messages):\n        # The GPT-4V format, where the content is an array of data\n        contents = message.get("content", [])\n        for content in contents:\n            if isinstance(content, str):\n                continue\n            if content.get("type", "") == "image_url":\n                img_data = content["image_url"]["url"]\n                images.append(img_utils.get_pil_image(img_data))\n\n    if not images:\n        raise ValueError("No image data found in messages.")\n\n    return images\n'})}),"\n",(0,a.jsx)(n.p,{children:"Start the converstion"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:'dalle = image_generator_agent()\ncritic = critic_agent()\n\nimg_prompt = "A happy dog wearing a shirt saying \'I Love AutoGen\'. Make sure the text is clear."\n# img_prompt = "Ask me how I\'m doing"\n\nresult = dalle.initiate_chat(critic, message=img_prompt)\n'})}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-text",children:"dalle (to critic):\n\nA happy dog wearing a shirt saying 'I Love AutoGen'. Make sure the text is clear.\n\n--------------------------------------------------------------------------------\ncritic (to dalle):\n\nCRITICS: the image needs to improve the contrast and size of the text to enhance its clarity, and the shirt's color should not clash with the dog's fur color to maintain a harmonious color scheme.\n\nPROMPT: here is the updated prompt!\nCreate an image of a joyful dog with a coat of a contrasting color to its fur, wearing a shirt with bold, large text saying 'I Love AutoGen' for clear readability.\n\n--------------------------------------------------------------------------------\ndalle (to critic):\n\nI generated an image with the prompt: Joyful dog, contrasting coat color to its fur, shirt with bold, large text \"I Love AutoGen\" for clear readability.<image>\n\n--------------------------------------------------------------------------------\ncritic (to dalle):\n\nCRITICS: the image effectively showcases a joyful dog with a contrasting shirt color, and the text 'I Love AutoGen' is large and bold, ensuring clear readability.\n\nPROMPT: TERMINATE\n\n--------------------------------------------------------------------------------\n"})}),"\n",(0,a.jsx)(n.p,{children:"Let\u2019s display all the images that was generated by Dalle"}),"\n",(0,a.jsx)(n.pre,{children:(0,a.jsx)(n.code,{className:"language-python",children:"images = extract_images(dalle, critic)\n\nfor image in reversed(images):\n    display(image.resize((300, 300)))\n"})}),"\n",(0,a.jsx)(n.p,{children:(0,a.jsx)(n.img,{src:t(83892).Z+"",width:"300",height:"300"})})]})}function m(e={}){const{wrapper:n}={...(0,i.a)(),...e.components};return n?(0,a.jsx)(n,{...e,children:(0,a.jsx)(c,{...e})}):c(e)}},83892:(e,n,t)=>{t.d(n,{Z:()=>a});const a=t.p+"assets/images/cell-8-output-1-abd77e50232f666b6e52602b1ce5b4d1.png"},11151:(e,n,t)=>{t.d(n,{Z:()=>r,a:()=>s});var a=t(67294);const i={},o=a.createContext(i);function s(e){const n=a.useContext(o);return a.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function r(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:s(e.components),a.createElement(o.Provider,{value:n},e.children)}}}]);