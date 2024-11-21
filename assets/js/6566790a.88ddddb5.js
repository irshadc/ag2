"use strict";(self.webpackChunkwebsite=self.webpackChunkwebsite||[]).push([[43755],{81278:(e,n,c)=>{c.r(n),c.d(n,{assets:()=>a,contentTitle:()=>r,default:()=>o,frontMatter:()=>t,metadata:()=>l,toc:()=>h});var s=c(85893),i=c(11151);const t={sidebar_label:"cache",title:"cache.cache"},r=void 0,l={id:"reference/cache/cache",title:"cache.cache",description:"Cache",source:"@site/docs/reference/cache/cache.md",sourceDirName:"reference/cache",slug:"/reference/cache/",permalink:"/ag2/docs/reference/cache/",draft:!1,unlisted:!1,editUrl:"https://github.com/ag2ai/ag2/edit/main/website/docs/reference/cache/cache.md",tags:[],version:"current",frontMatter:{sidebar_label:"cache",title:"cache.cache"},sidebar:"referenceSideBar",previous:{title:"abstract_cache_base",permalink:"/ag2/docs/reference/cache/abstract_cache_base"},next:{title:"cache_factory",permalink:"/ag2/docs/reference/cache/cache_factory"}},a={},h=[{value:"Cache",id:"cache",level:2},{value:"redis",id:"redis",level:3},{value:"disk",id:"disk",level:3},{value:"cosmos_db",id:"cosmos_db",level:3},{value:"__init__",id:"__init__",level:3},{value:"__enter__",id:"__enter__",level:3},{value:"__exit__",id:"__exit__",level:3},{value:"get",id:"get",level:3},{value:"set",id:"set",level:3},{value:"close",id:"close",level:3}];function d(e){const n={code:"code",em:"em",h2:"h2",h3:"h3",li:"li",p:"p",pre:"pre",strong:"strong",ul:"ul",...(0,i.a)(),...e.components};return(0,s.jsxs)(s.Fragment,{children:[(0,s.jsx)(n.h2,{id:"cache",children:"Cache"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"class Cache(AbstractCache)\n"})}),"\n",(0,s.jsx)(n.p,{children:"A wrapper class for managing cache configuration and instances."}),"\n",(0,s.jsx)(n.p,{children:"This class provides a unified interface for creating and interacting with\ndifferent types of cache (e.g., Redis, Disk). It abstracts the underlying\ncache implementation details, providing methods for cache operations."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Attributes"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"config"})," ",(0,s.jsx)(n.em,{children:"Dict[str, Any]"})," - A dictionary containing cache configuration."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"cache"})," - The cache instance created based on the provided configuration."]}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"redis",children:"redis"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'@staticmethod\ndef redis(cache_seed: Union[str, int] = 42,\n          redis_url: str = "redis://localhost:6379/0") -> "Cache"\n'})}),"\n",(0,s.jsx)(n.p,{children:"Create a Redis cache instance."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"cache_seed"})," ",(0,s.jsx)(n.em,{children:"Union[str, int], optional"})," - A seed for the cache. Defaults to 42."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"redis_url"})," ",(0,s.jsx)(n.em,{children:"str, optional"}),' - The URL for the Redis server. Defaults to "redis://localhost:6379/0".']}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"Cache"})," - A Cache instance configured for Redis."]}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"disk",children:"disk"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'@staticmethod\ndef disk(cache_seed: Union[str, int] = 42,\n         cache_path_root: str = ".cache") -> "Cache"\n'})}),"\n",(0,s.jsx)(n.p,{children:"Create a Disk cache instance."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"cache_seed"})," ",(0,s.jsx)(n.em,{children:"Union[str, int], optional"})," - A seed for the cache. Defaults to 42."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"cache_path_root"})," ",(0,s.jsx)(n.em,{children:"str, optional"}),' - The root path for the disk cache. Defaults to ".cache".']}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"Cache"})," - A Cache instance configured for Disk caching."]}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"cosmos_db",children:"cosmos_db"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'@staticmethod\ndef cosmos_db(connection_string: Optional[str] = None,\n              container_id: Optional[str] = None,\n              cache_seed: Union[str, int] = 42,\n              client: Optional[any] = None) -> "Cache"\n'})}),"\n",(0,s.jsx)(n.p,{children:"Create a Cosmos DB cache instance with 'autogen_cache' as database ID."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"connection_string"})," ",(0,s.jsx)(n.em,{children:"str, optional"})," - Connection string to the Cosmos DB account."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"container_id"})," ",(0,s.jsx)(n.em,{children:"str, optional"})," - The container ID for the Cosmos DB account."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"cache_seed"})," ",(0,s.jsx)(n.em,{children:"Union[str, int], optional"})," - A seed for the cache."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"client"})," - Optional[CosmosClient]: Pass an existing Cosmos DB client."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"Cache"})," - A Cache instance configured for Cosmos DB."]}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"__init__",children:"__init__"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def __init__(config: Dict[str, Any])\n"})}),"\n",(0,s.jsx)(n.p,{children:"Initialize the Cache with the given configuration."}),"\n",(0,s.jsx)(n.p,{children:"Validates the configuration keys and creates the cache instance."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"config"})," ",(0,s.jsx)(n.em,{children:"Dict[str, Any]"})," - A dictionary containing the cache configuration."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Raises"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"ValueError"})," - If an invalid configuration key is provided."]}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"__enter__",children:"__enter__"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:'def __enter__() -> "Cache"\n'})}),"\n",(0,s.jsx)(n.p,{children:"Enter the runtime context related to the cache object."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsx)(n.p,{children:"The cache instance for use within a context block."}),"\n",(0,s.jsx)(n.h3,{id:"__exit__",children:"__exit__"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def __exit__(exc_type: Optional[Type[BaseException]],\n             exc_value: Optional[BaseException],\n             traceback: Optional[TracebackType]) -> None\n"})}),"\n",(0,s.jsx)(n.p,{children:"Exit the runtime context related to the cache object."}),"\n",(0,s.jsx)(n.p,{children:"Cleans up the cache instance and handles any exceptions that occurred\nwithin the context."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"exc_type"})," - The exception type if an exception was raised in the context."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"exc_value"})," - The exception value if an exception was raised in the context."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"traceback"})," - The traceback if an exception was raised in the context."]}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"get",children:"get"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def get(key: str, default: Optional[Any] = None) -> Optional[Any]\n"})}),"\n",(0,s.jsx)(n.p,{children:"Retrieve an item from the cache."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"key"})," ",(0,s.jsx)(n.em,{children:"str"})," - The key identifying the item in the cache."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"default"})," ",(0,s.jsx)(n.em,{children:"optional"})," - The default value to return if the key is not found.\nDefaults to None."]}),"\n"]}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Returns"}),":"]}),"\n",(0,s.jsx)(n.p,{children:"The value associated with the key if found, else the default value."}),"\n",(0,s.jsx)(n.h3,{id:"set",children:"set"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def set(key: str, value: Any) -> None\n"})}),"\n",(0,s.jsx)(n.p,{children:"Set an item in the cache."}),"\n",(0,s.jsxs)(n.p,{children:[(0,s.jsx)(n.strong,{children:"Arguments"}),":"]}),"\n",(0,s.jsxs)(n.ul,{children:["\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"key"})," ",(0,s.jsx)(n.em,{children:"str"})," - The key under which the item is to be stored."]}),"\n",(0,s.jsxs)(n.li,{children:[(0,s.jsx)(n.code,{children:"value"})," - The value to be stored in the cache."]}),"\n"]}),"\n",(0,s.jsx)(n.h3,{id:"close",children:"close"}),"\n",(0,s.jsx)(n.pre,{children:(0,s.jsx)(n.code,{className:"language-python",children:"def close() -> None\n"})}),"\n",(0,s.jsx)(n.p,{children:"Close the cache."}),"\n",(0,s.jsx)(n.p,{children:"Perform any necessary cleanup, such as closing connections or releasing resources."})]})}function o(e={}){const{wrapper:n}={...(0,i.a)(),...e.components};return n?(0,s.jsx)(n,{...e,children:(0,s.jsx)(d,{...e})}):d(e)}},11151:(e,n,c)=>{c.d(n,{Z:()=>l,a:()=>r});var s=c(67294);const i={},t=s.createContext(i);function r(e){const n=s.useContext(t);return s.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function l(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(i):e.components||i:r(e.components),s.createElement(t.Provider,{value:n},e.children)}}}]);