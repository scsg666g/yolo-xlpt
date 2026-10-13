import type { CSSProperties } from 'react'
import { LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import ReactEChartsCore from 'echarts-for-react/lib/core'

echarts.use([LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

export function LineChartView({ option, style }: { option: unknown; style?: CSSProperties }) {
  return <ReactEChartsCore echarts={echarts} option={option} style={style} />
}
