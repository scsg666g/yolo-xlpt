import { useMemo, type CSSProperties } from 'react'
import { LineChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import * as echarts from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import { useAppearance } from '../appearance'

echarts.use([LineChart, GridComponent, LegendComponent, TooltipComponent, CanvasRenderer])

type ChartSection = Record<string, unknown>

function section(value: unknown): ChartSection {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as ChartSection : {}
}

function mapSections(value: unknown, update: (value: ChartSection) => ChartSection): unknown {
  if (value == null) return value
  return Array.isArray(value) ? value.map((item) => update(section(item))) : update(section(value))
}

function themedOption(option: unknown, theme: 'light' | 'dark') {
  const source = section(option)
  const colors = theme === 'light'
    ? { text: '#294454', muted: '#637887', line: '#bcced9', grid: '#dce6ed', tooltip: '#ffffff' }
    : { text: '#e8eff4', muted: '#96a6b2', line: '#344653', grid: '#2a3946', tooltip: '#18242f' }
  const axis = (value: ChartSection): ChartSection => ({
    ...value,
    nameTextStyle: { ...section(value.nameTextStyle), color: colors.muted },
    axisLabel: { ...section(value.axisLabel), color: colors.muted },
    axisLine: { ...section(value.axisLine), lineStyle: { ...section(section(value.axisLine).lineStyle), color: colors.line } },
    axisTick: { ...section(value.axisTick), lineStyle: { ...section(section(value.axisTick).lineStyle), color: colors.line } },
    splitLine: { ...section(value.splitLine), lineStyle: { ...section(section(value.splitLine).lineStyle), color: colors.grid } },
  })
  // Theme only chart chrome; experiment and metric colors remain independent
  // of the user's interface accent, as do data, formatting, and animation.
  return {
    ...source,
    textStyle: { ...section(source.textStyle), color: colors.text },
    xAxis: mapSections(source.xAxis, axis),
    yAxis: mapSections(source.yAxis, axis),
    legend: mapSections(source.legend, (value) => ({
      ...value,
      textStyle: { ...section(value.textStyle), color: colors.muted },
      pageTextStyle: { ...section(value.pageTextStyle), color: colors.muted },
      pageIconColor: colors.muted,
      pageIconInactiveColor: colors.line,
    })),
    tooltip: mapSections(source.tooltip, (value) => ({
      ...value,
      backgroundColor: colors.tooltip,
      borderColor: colors.line,
      textStyle: { ...section(value.textStyle), color: colors.text },
    })),
  }
}

export function LineChartView({ option, style }: { option: unknown; style?: CSSProperties }) {
  const { resolvedTheme } = useAppearance()
  const chartOption = useMemo(() => themedOption(option, resolvedTheme), [option, resolvedTheme])
  return <ReactEChartsCore echarts={echarts} option={chartOption} style={style} />
}
