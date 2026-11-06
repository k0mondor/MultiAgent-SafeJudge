import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react'
import {ReactFlow, ReactFlowProvider, Handle, Position, MarkerType, useReactFlow, useViewport, type Node, type NodeProps} from '@xyflow/react'
import {ArrowLeft, ArrowRight, Maximize, Minus, Plus, ArrowUpRight} from 'lucide-react'
import {Button} from '@/components/ui/button'
import {Sheet, SheetContent, SheetHeader, SheetTitle, SheetDescription} from '@/components/ui/sheet'
import {Tabs, TabsList, TabsTrigger, TabsContent} from '@/components/ui/tabs'
import {catalog, stages, defaults, type NodeInfo, type Role} from './flow-data'
import registry from './registry.json'
import {FlowEdge} from './FlowEdge'
import Cases from './Cases'
import '@xyflow/react/dist/style.css'
import './styles.css'

const base=new URL('.',document.baseURI).pathname
const routePath=()=>location.pathname.startsWith(base)?'/'+location.pathname.slice(base.length):location.pathname
const shortModel=(profile:string)=>{
 const id=registry.models.find(m=>m.profile_id===profile)?.model_id||profile
 return ({'deepseek/deepseek-v4-flash':'DeepSeek V4 Flash','z-ai/glm-4.6v':'GLM-4.6V','bytedance-seed/seed-2.0-mini':'Seed 2.0 Mini','meta-llama/llama-guard-4-12b':'Llama Guard 4 12B'} as Record<string,string>)[id]||id.split('/').at(-1)!
}
type FlowData=NodeInfo & {model?:string;open:()=>void;focused:boolean;arriving:boolean}
function WorkflowNode({data}:NodeProps<Node<FlowData>>){return <div className={`workflow-node ${data.focused?'is-focused':''} ${data.arriving?'is-arriving':''}`}>
 <Handle type="target" position={Position.Left} id="left"/><Handle type="target" position={Position.Top} id="top"/><Handle type="target" position={Position.Bottom} id="bottom"/>
 <button className="node-body nodrag" onClick={data.open} aria-label={`查看${data.title}节点`}>
 <div className="node-title"><strong>{data.title}</strong>{data.kind&&<span className="node-kind">{data.kind}</span>}</div>
 {data.model&&<div className="node-model"><span>模型</span><span>{data.model}</span></div>}
 <p>{data.summary}</p></button>
 {data.role&&<button className="node-action nodrag" onClick={data.open}>查看节点<ArrowUpRight size={13}/></button>}
 <Handle type="source" position={Position.Right} id="right"/><Handle type="source" position={Position.Bottom} id="out-bottom"/>
 </div>}
const nodeTypes={workflow:WorkflowNode}
const edgeTypes={flow:FlowEdge}
function OverviewRegions({onEnter}:{onEnter:(stage:number)=>void}){
 const {x,y,zoom}=useViewport()
 const visible=zoom<0.55
 return <div className="overview-regions" style={{opacity:visible?1:0}} aria-hidden={!visible}>
  {stages.map((stage,i)=>{
   const left=i*1660-65, top=-85
   const width=Math.max(...stage.nodes.map(node=>node[1]))+410
   const height=Math.max(...stage.nodes.map(node=>node[2]))+300
   return <button key={stage.title} className="overview-region nodrag nopan" aria-label={`进入${stage.title}微观视图`} tabIndex={visible?0:-1} onClick={()=>onEnter(i)} style={{left:x+left*zoom,top:y+top*zoom-37,width:width*zoom,height:height*zoom+37,pointerEvents:visible?'auto':'none'}}>
    <span className="overview-region-label" style={{pointerEvents:visible?'auto':'none'}} onClick={event=>{event.stopPropagation();onEnter(i)}}>
     <span>0{i+1}</span><strong>{stage.title}</strong><ArrowUpRight size={14}/>
    </span>
   </button>
  })}
 </div>
}
const cameraEase=(t:number)=>t<0.5?4*t*t*t:1-Math.pow(-2*t+2,3)/2
export default function App(){
 const [route,setRoute]=useState(routePath()==='/cases'?'cases':'flow')
 const models=defaults
 const policy='1.3'
 const navigate=useCallback((next:string)=>{history.pushState({},'',next==='flow'?base:`${base}${next}`);setRoute(next);window.scrollTo(0,0)},[])
 useEffect(()=>{const pop=()=>setRoute(routePath()==='/cases'?'cases':'flow');window.addEventListener('popstate',pop);return()=>window.removeEventListener('popstate',pop)},[])
 useEffect(()=>{document.title=`SafeJudge · ${{flow:'评估流程',cases:'案例展示'}[route]}`},[route])
 return <><header className="site-header"><a className="wordmark" href={base} onClick={e=>{e.preventDefault();navigate('flow')}}>SafeJudge</a><span className="project-name">MultiAgent-SafeJudge</span><nav aria-label="主导航">{[['flow','评估流程'],['cases','案例展示']].map(([r,label])=><a key={r} href={r==='flow'?base:`${base}${r}`} aria-current={route===r?'page':undefined} onClick={e=>{if(!e.ctrlKey&&!e.metaKey){e.preventDefault();navigate(r)}}}>{label}</a>)}</nav></header>
 {route==='flow'?<ReactFlowProvider><Explorer models={models} policy={policy}/></ReactFlowProvider>:<Cases/>}
 </>
}
function Explorer({models,policy}:{models:Record<Role,string>;policy:string}){
 const [stage,setStage]=useState(0),[selected,setSelected]=useState<string|null>(null),[overview,setOverview]=useState(false),[zoom,setZoom]=useState(1)
 const {fitView,getViewport,setViewport,zoomTo}=useReactFlow()
 const [focusOpen,setFocusOpen]=useState(false)
 const [steps,setSteps]=useState([0,0,0,0])
 const [arrivals,setArrivals]=useState<string[]>([])
 const returnView=useRef({x:0,y:0,zoom:1})
 const closeTimer=useRef<ReturnType<typeof setTimeout>|undefined>(undefined)
 const stageTimer=useRef<ReturnType<typeof setTimeout>|undefined>(undefined)
 const trigger=useRef<HTMLElement|null>(null)
 const [reducedMotion,setReducedMotion]=useState(false)
 const reduce=useRef(false)
 reduce.current=reducedMotion
 useEffect(()=>{document.body.classList.toggle('reduced-motion',reducedMotion);return()=>document.body.classList.remove('reduced-motion')},[reducedMotion])
 const canvas=useRef<HTMLDivElement>(null)
 const current=stages[stage],selection=selected?catalog[selected]:null
 const zoomTarget=useRef<number|null>(null)
 const zoomButtonTimer=useRef<ReturnType<typeof setTimeout>|undefined>(undefined)
 const changeZoom=(direction:number)=>{
  const base=zoomTarget.current??getViewport().zoom
  const next=Math.max(0.12,Math.min(2.5,direction>0?base*1.6:base/1.6))
  zoomTarget.current=next
  clearTimeout(zoomButtonTimer.current)
  void zoomTo(next,{duration:reduce.current?0:220})
  zoomButtonTimer.current=setTimeout(()=>{zoomTarget.current=null},240)
 }
 useEffect(()=>()=>clearTimeout(zoomButtonTimer.current),[])
 const fit=useCallback(()=>{void fitView({padding:0.12,duration:reduce.current?0:450,maxZoom:1,minZoom:overview?0.12:0.85})},[fitView,overview])
 const open=useCallback((id:string,x:number,y:number)=>{
  clearTimeout(closeTimer.current)
  if(!selected){returnView.current=getViewport();trigger.current=document.activeElement as HTMLElement}
  setSelected(id);setFocusOpen(true)
  const width=canvas.current?.clientWidth||1200,height=canvas.current?.clientHeight||700
  const targetZoom=Math.min(1.35,Math.max(1.18,returnView.current.zoom*1.25))
  void setViewport({x:(width-440)/2-(x+140)*targetZoom,y:height/2-(y+80)*targetZoom,zoom:targetZoom},
    {duration:reduce.current?0:780,ease:cameraEase,interpolate:'linear'})
 },[getViewport,setViewport,selected])
 const closeFocus=useCallback(()=>{
  if(!focusOpen)return
  setFocusOpen(false)
  void setViewport(returnView.current,{duration:reduce.current?0:420,ease:cameraEase,interpolate:'linear'})
  closeTimer.current=setTimeout(()=>{setSelected(null);trigger.current?.focus({preventScroll:true})},reduce.current?0:440)
 },[focusOpen,setViewport])
 useEffect(()=>{document.body.classList.toggle('node-focus-active',focusOpen);return()=>document.body.classList.remove('node-focus-active')},[focusOpen])
 useEffect(()=>()=>{clearTimeout(closeTimer.current);clearTimeout(stageTimer.current)},[])

 const nodes=useMemo(()=>{
 const groups=overview?stages.map((s,i)=>({s,i})): [{s:current,i:stage}]
 return groups.flatMap(({s,i})=>s.nodes.map(([id,x,y])=>({id:`${i}-${id}`,type:'workflow',className:selected===id?'focus-node':'',zIndex:selected===id?20:0,position:{x:x+(overview?i*1660:0),y},data:{...catalog[id],model:catalog[id].role?shortModel(models[catalog[id].role!]):undefined,focused:selected===id&&focusOpen,arriving:!overview&&arrivals.includes(id),open:()=>{if(overview){setOverview(false);setStage(i);setSelected(null)}else open(id,x,y)}}})))
 },[stage,current,overview,models,selected,focusOpen,open,arrivals])
 const edges=useMemo(()=>{const groups=overview?stages.map((s,i)=>({s,i})):[{s:current,i:stage}];const all=groups.flatMap(({s,i})=>s.edges.map(([a,b,label,kind],j)=>({id:`${i}-edge-${j}`,source:`${i}-${a}`,target:`${i}-${b}`,sourceHandle:kind==='vertical'?'out-bottom':'right',targetHandle:kind==='vertical'?'top':kind==='bottom'?'bottom':'left',type:kind==='bottom'?'smoothstep':'default',label,markerEnd:{type:MarkerType.ArrowClosed,color:'#858b96',width:16,height:16},style:{stroke:'#858b96',strokeWidth:1.3},labelStyle:{fill:'#5d626c',fontSize:12},labelBgStyle:{fill:'#f2f3f5'},labelBgPadding:[7,4] as [number,number]})));if(overview){for(let i=0;i<3;i++){const sources=['input','routing','verdicts'];const targets=['request','facts','verdicts'];all.push({id:`bridge-${i}`,source:`${i}-${sources[i]}`,target:`${i+1}-${targets[i]}`,sourceHandle:'right',targetHandle:'left',type:'default',label:'下一阶段',markerEnd:{type:MarkerType.ArrowClosed,color:'#858b96',width:16,height:16},style:{stroke:'#858b96',strokeWidth:1.3},labelStyle:{fill:'#5d626c',fontSize:12},labelBgStyle:{fill:'#f2f3f5'},labelBgPadding:[7,4]})}}return all.map(edge=>({...edge,type:'flow',data:{orthogonal:edge.type==='smoothstep',progress:overview?0:steps[stage],muted:focusOpen&&!edge.source.endsWith(`-${selected}`)&&!edge.target.endsWith(`-${selected}`)}}))},[overview,current,stage,steps,focusOpen,selected])
 const changeStage=useCallback((next:number)=>{clearTimeout(stageTimer.current);clearTimeout(closeTimer.current);setArrivals([]);setFocusOpen(false);setSelected(null);setOverview(false);setStage(Math.max(0,Math.min(3,next)))},[])
 useEffect(()=>{const timer=setTimeout(fit,80);return()=>clearTimeout(timer)},[stage,overview,fit])

 useEffect(()=>{
  const el=canvas.current;if(!el)return
  const handle=(e:WheelEvent)=>{
   if(selected)return
   e.preventDefault();e.stopPropagation()
   clearTimeout(zoomButtonTimer.current);zoomTarget.current=null
   const view=getViewport(),rect=el.getBoundingClientRect()
   const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?500:1)
   const next=Math.max(0.12,Math.min(2.5,view.zoom*Math.exp(-Math.max(-160,Math.min(160,delta))*(e.ctrlKey?0.012:0.006))))
   const x=e.clientX-rect.left,y=e.clientY-rect.top,ratio=next/view.zoom
   void setViewport({x:x-(x-view.x)*ratio,y:y-(y-view.y)*ratio,zoom:next})
  }
  el.addEventListener('wheel',handle,{passive:false,capture:true});return()=>el.removeEventListener('wheel',handle,true)
 },[selected,getViewport,setViewport])
 // Stage entry owns the reveal; camera movement and focus never replay it.
 useEffect(()=>{
  setSteps([0,0,0,0]);setArrivals([])
  if(overview)return
  const start=setTimeout(()=>{
   setSteps(prev=>prev.map((_,i)=>i===stage?1:0))
   stageTimer.current=setTimeout(()=>setArrivals([...new Set(current.edges.map(edge=>edge[1]))]),reduce.current?0:700)
  },reduce.current?0:550)
  return()=>{clearTimeout(start);clearTimeout(stageTimer.current)}
 },[stage,overview,current])
 const policyData=registry.policies.find(p=>p.aggregator_version===policy)!
 return <main className="explorer"><div className="page-heading"><div><h1>评估流程</h1><p>节点职责与数据传递</p></div><Button variant="outline" onClick={()=>{setSelected(null);setOverview(!overview)}}><Maximize/>{overview?'返回当前阶段':'查看全流程'}</Button></div>
 <div className="stage-bar"><nav aria-label="流程阶段">{stages.map((s,i)=><button key={s.title} aria-current={!overview&&stage===i?'step':undefined} onClick={()=>changeStage(i)}><span>0{i+1}</span>{s.title}</button>)}</nav><div className="stage-arrows"><Button variant="outline" size="icon" aria-label="上一阶段" disabled={stage===0} onClick={()=>changeStage(stage-1)}><ArrowLeft/></Button><Button variant="outline" size="icon" aria-label="下一阶段" disabled={stage===3} onClick={()=>changeStage(stage+1)}><ArrowRight/></Button></div></div>
 <div className="canvas" ref={canvas}><div className="canvas-heading" aria-live="polite"><h2>{overview?'完整评估流程':current.title}</h2><span>{overview?'点击分区进入对应阶段':`阶段 0${stage+1} / 04`}</span></div>

 <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} edgeTypes={edgeTypes} nodesDraggable={false} nodesConnectable={false} elementsSelectable={false} fitView fitViewOptions={{padding:0.12,maxZoom:1,minZoom:0.85}} minZoom={0.12} maxZoom={2.5} zoomOnScroll={false} zoomOnPinch={false} zoomActivationKeyCode={null} zoomOnDoubleClick={overview} panOnScroll={false} preventScrolling={overview} onMove={(_,v)=>setZoom(v.zoom)} onPaneClick={closeFocus} proOptions={{hideAttribution:false}} aria-label="安全评估流程画布">
 {overview&&<OverviewRegions onEnter={changeStage}/>}
 </ReactFlow>
 <div className="canvas-controls"><Button variant="ghost" size="icon" aria-label="放大画布" onClick={()=>changeZoom(1)}><Plus/></Button><Button variant="ghost" size="icon" aria-label="缩小画布" onClick={()=>changeZoom(-1)}><Minus/></Button><span>{Math.round(zoom*100)}%</span><Button variant="ghost" size="icon" aria-label="适应画布" onClick={fit}><Maximize/></Button></div><Button className="motion-control" variant="outline" aria-pressed={reducedMotion} onClick={()=>setReducedMotion(v=>!v)}>{reducedMotion?'恢复动效':'减少动效'}</Button></div>
 <Sheet open={focusOpen} onOpenChange={isOpen=>{if(!isOpen)closeFocus()}}><SheetContent className="node-sheet" onCloseAutoFocus={e=>e.preventDefault()}><SheetHeader><SheetTitle>{selection?.title}</SheetTitle><SheetDescription>{selection?.summary}</SheetDescription></SheetHeader>{selection&&<Tabs defaultValue="function" key={selection.id}><TabsList><TabsTrigger value="function">功能</TabsTrigger><TabsTrigger value="io">输入与输出</TabsTrigger><TabsTrigger value="model">{selection.role?'模型':'规则与来源'}</TabsTrigger></TabsList><TabsContent value="function"><div className="detail-copy">{selection.details.map((s,i)=><div key={s}><span>{String(i+1).padStart(2,'0')}</span><p>{s}</p></div>)}</div>{selection.id==='aggregate'&&<div className="formula"><h3>当前策略 v{policy}</h3><code>V × (S+1) × (C+1) × (F+1) × (1+0.5E)</code><p>L0：0<br/>L1：0 &lt; 分数 &lt; {policyData.l2_threshold}<br/>L2：分数 ≥ {policyData.l2_threshold}</p></div>}</TabsContent><TabsContent value="io"><div className="io-section"><h3>输入</h3>{selection.inputs.map(s=><p key={s}>{s}</p>)}</div><div className="io-section"><h3>输出</h3>{selection.outputs.map(s=><p key={s}>{s}</p>)}</div></TabsContent><TabsContent value="model">{selection.role?<div className="model-detail"><h3>{shortModel(models[selection.role])}</h3><code>{models[selection.role]}</code><p>模型配置仅用于本地展示。</p></div>:<p className="rules-copy">由确定性代码或数据协议执行，不调用模型。</p>}<div className="source-block"><h3>实现来源</h3><code>{selection.source}</code></div></TabsContent></Tabs>}</SheetContent></Sheet>
 </main>
}
