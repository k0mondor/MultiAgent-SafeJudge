import { BaseEdge, EdgeText, getBezierPath, getSmoothStepPath, type EdgeProps } from '@xyflow/react'

/** Scroll reveals the actual connection path; it never represents inference status. */
export function FlowEdge(props: EdgeProps) {
  const { id, data, label, markerEnd, style } = props
  const progress = Number(data?.progress ?? 0)
  const [path, labelX, labelY] = data?.orthogonal
    ? getSmoothStepPath({ ...props, borderRadius: 20 })
    : getBezierPath(props)
  const length = progress * 100
  return <g className={`flow-connection ${data?.muted ? 'is-muted' : ''}`}>
    <BaseEdge id={id} path={path} markerEnd={markerEnd} style={style} />
    <path className="flow-reveal" d={path} pathLength={100} fill="none"
      strokeDasharray={`${length} 100`}
      strokeDashoffset={0} style={{strokeLinecap: progress > 0 ? 'round' : 'butt'}}
      aria-hidden="true" />
    {label && <EdgeText x={labelX} y={labelY} label={label}
      labelStyle={{ fill: '#5d626c', fontSize: 12 }}
      labelBgStyle={{ fill: '#f2f3f5' }} labelBgPadding={[7, 4]} />}
  </g>
}
