import {useState} from 'react'
import {ChevronDown, Maximize2} from 'lucide-react'
import cases from './cases-data.json'
import './cases.css'

type Case = typeof cases[number]
const axes=[['specificity','S','具体性'],['completeness','C','完整性'],['feasibility','F','可行性'],['evasion_help','E','规避帮助']] as const
function Highlight({text,match}:{text:string;match:string}){const index=text.indexOf(match);return index<0?<>{text}</>:<>{text.slice(0,index)}<mark>{match}</mark>{text.slice(index+match.length)}</>}
function RecordBlock({title,value}:{title:string;value:unknown}){return <details className="record-block"><summary>{title}<ChevronDown size={15}/></summary><pre>{JSON.stringify(value,null,2)}</pre></details>}
function CaseDetail({item}:{item:Case}){
 const [category,setCategory]=useState(0)
 const result=item.results[category]
 const initial=result?.verdicts.find(v=>v.axis==='harm_enablement')?.enablement_scores
 const arbitration=result?.arbitration
 const scores=arbitration?.reconciled_enablement_scores??initial
 const guard=result?.guardrail_verdict
 const compliance=result?.verdicts.find(v=>v.axis==='compliance')?.label
 return <div className="case-expanded">
  <dl className="case-identity"><div><dt>数据集</dt><dd>{item.dataset}</dd></div><div><dt>被测模型</dt><dd>{item.targetModel}</dd></div><div><dt>案例 ID</dt><dd>{item.id}</dd></div></dl>
  <div className="case-columns"><section className="case-reading"><h3>请求</h3>
   {item.media.map(media=>media.type==='image'?<a className="case-image" href={media.url} target="_blank" rel="noreferrer" key={media.url} aria-label="打开原始图片"><img src={media.url} alt={`${item.title}的原始输入图片`}/><Maximize2 size={16}/></a>:<audio key={media.url} controls preload="metadata" src={media.url} aria-label="原始输入音频">浏览器不支持音频播放。<a href={media.url}>下载原始音频</a></audio>)}
   <p className="case-request">{item.requestZh}</p>
   <details className="case-original"><summary>原始请求</summary><p>{item.request}</p></details>
   <h3 className="case-answer-heading">回答 <span>译文 · 节选</span></h3>
   <p className="case-answer"><Highlight text={item.answerZh} match={item.highlight}/></p>
   <details className="case-full"><summary>完整回答<ChevronDown size={14}/></summary><p><Highlight text={item.response} match={item.evidence}/></p></details>
  </section>
  <aside className="case-verdict"><h3>结论</h3><div className="case-level">L{item.aggregate.response_compliance_level}</div><p>{item.conclusion}</p>
   {item.aggregate.oversensitive===true&&<section><h3>过度拒绝</h3><p>是</p><p className="case-note">请求限定为风险讨论，回答开头仍拒绝高层次讨论；后续也提供了部分风险说明，可结合全文复核。</p></section>}
   {result&&<>{item.results.length>1&&<select aria-label="风险小类" value={category} onChange={e=>setCategory(Number(e.target.value))}>{item.results.map((r,i)=><option value={i} key={r.category_id}>{r.category_id} · {r.category_name}</option>)}</select>}
   <section><h3>Llama Guard 4</h3><div className="case-value"><span>原生判断</span><strong>{guard?.native_label??'无记录'}</strong></div><div className="case-value"><span>合规判断</span><strong>{compliance==='compliant'?'合规':compliance==='noncompliant'?'不合规':'无记录'}</strong></div></section>
   {scores&&<section><h3>四轴评分</h3><table className="axis-table"><thead><tr><th>维度</th>{arbitration&&<th>原始</th>}<th>{arbitration?'仲裁后':'评分'}</th></tr></thead><tbody>{axes.map(([key,code,label])=><tr key={key}><td><span>{code}</span>{label}</td>{arbitration&&<td>{initial?.[key]??'—'}</td>}<td>{scores[key]}</td></tr>)}</tbody></table></section>}
   {arbitration&&<details className="case-full"><summary>仲裁记录<ChevronDown size={14}/></summary><p>合规结论与危害评分存在冲突。本次仲裁修正危害评分，再由规则聚合为 L{item.aggregate.response_compliance_level}。</p><RecordBlock title="仲裁原始输出" value={arbitration}/></details>}</>}
  </aside></div>
  <details className="case-fold"><summary>中间输出<ChevronDown size={17}/></summary><div className="case-intermediate">
   <section><h3>意图分析 · Intent</h3><dl><dt>请求意图</dt><dd>{{benign:'良性',harmful:'有害',ambiguous:'不明确'}[item.intent.request_intent]??item.intent.request_intent}</dd><dt>分析状态</dt><dd>{item.intent.analysis_status}</dd><dt>依据</dt><dd>{item.intent.intent_basis}</dd><dt>请求行为</dt><dd>{item.intent.requested_action}</dd></dl><RecordBlock title="完整 Intent 输出" value={item.intent}/></section>
   <section><h3>风险识别与路由</h3><dl><dt>请求侧类别</dt><dd>{item.categories.request_category_ids?.join(' · ')||'无'}</dd><dt>回答新增类别</dt><dd>{item.categories.response_added_category_ids?.join(' · ')||'无'}</dd><dt>裁判小类</dt><dd>{item.results.map(r=>`${r.category_id} ${r.category_name}`).join('；')||'请求级判断'}</dd></dl><RecordBlock title="类别原始输出" value={item.categories}/></section>
   {item.grounding.observations.length>0&&<section className="case-grounding"><h3>媒体理解</h3>{item.grounding.observations.map((o,i)=><p key={i}>{o.text}</p>)}<RecordBlock title="媒体理解原始输出" value={item.grounding}/></section>}
   <section><h3>裁判与聚合</h3><RecordBlock title="裁判原始输出" value={{categories:item.results,requestVerdicts:item.verdicts}}/><RecordBlock title="聚合原始输出" value={item.aggregate}/></section>
  </div></details>
  <details className="case-fold"><summary>实验信息<ChevronDown size={17}/></summary><dl className="case-provenance"><dt>案例 ID</dt><dd>{item.id}</dd><dt>数据集版本</dt><dd>{item.provenance.datasetVersion} · {item.provenance.split}</dd><dt>源样本 ID</dt><dd>{item.provenance.originalId}</dd><dt>回答 ID</dt><dd>{item.provenance.responseId}</dd><dt>评估标识</dt><dd>{item.provenance.evaluationKey}</dd><dt>回答 SHA-256</dt><dd>{item.provenance.responseHash}</dd><dt>裁判组 ID</dt><dd>{item.provenance.juryId}</dd><dt>被测模型</dt><dd>{item.targetModel}</dd><dt>协调与仲裁</dt><dd>{item.jury.coordinator.model}</dd><dt>四轴裁判配置</dt><dd>{item.jury.enablement_judge.model}</dd><dt>策略版本</dt><dd>v{item.policy}</dd><dt>数据来源</dt><dd>{item.dataset} · {item.license}</dd><dt>评估记录</dt><dd>{item.source}</dd>{item.media.map(media=><div className="case-media-provenance" key={media.url}><dt>媒体 SHA-256</dt><dd>{media.sha256}</dd></div>)}<dt>展示说明</dt><dd>中文请求与回答为展示译文或摘要，高亮为人工选取；中间输出保留日志原文。本次实验模型不随配置页切换。</dd></dl></details>
 </div>
}
export default function Cases(){const [active,setActive]=useState<string|null>(cases[0].id);return <main className="case-library"><h1>案例展示</h1><div className="case-list">{cases.map(item=><article className="case-item" key={item.id}><h2><button className="case-toggle" aria-expanded={active===item.id} aria-controls={`case-${item.id}`} onClick={()=>setActive(active===item.id?null:item.id)}><span>{item.title}</span><span className="case-type">{item.modality}</span><ChevronDown size={18}/></button></h2>{active===item.id&&<div id={`case-${item.id}`}><CaseDetail item={item}/></div>}</article>)}</div></main>}
