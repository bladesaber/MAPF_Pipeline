// @ts-ignore
import Plotly from 'plotly.js-dist'

export class GraphUtils {
  /**
   * Draw a line graph using Plotly
   */
  static drawLineGraph(
    containerId: string,
    xData: number[],
    yData: number[],
    title: string = 'Line Graph',
    xLabel: string = 'X Axis',
    yLabel: string = 'Y Axis',
    color: string = '#1f77b4'
  ): void {
    const trace = {
      x: xData,
      y: yData,
      type: 'scatter',
      mode: 'lines+markers',
      line: {
        color: color,
        width: 2
      },
      marker: {
        size: 4,
        color: color
      },
      name: 'Data'
    }

    const layout = {
      title: {
        text: title,
        font: {
          size: 16,
          color: '#333'
        }
      },
      xaxis: {
        title: {
          text: xLabel,
          font: {
            size: 12,
            color: '#666'
          }
        },
        gridcolor: '#f0f0f0',
        zeroline: false
      },
      yaxis: {
        title: {
          text: yLabel,
          font: {
            size: 12,
            color: '#666'
          }
        },
        gridcolor: '#f0f0f0',
        zeroline: false
      },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      margin: {
        l: 60,
        r: 40,
        t: 60,
        b: 60
      },
      showlegend: false
    }

    const config = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
    }

    Plotly.newPlot(containerId, [trace], layout, config)
  }

  /**
   * Draw a pie chart using Plotly
   */
  static drawPieChart(
    containerId: string,
    labels: string[],
    values: number[],
    title: string = 'Pie Chart',
    colors: string[] = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
  ): void {
    const trace = {
      labels: labels,
      values: values,
      type: 'pie',
      marker: {
        colors: colors.slice(0, labels.length)
      },
      textinfo: 'label+percent',
      textposition: 'outside',
      automargin: true
    }

    const layout = {
      title: {
        text: title,
        font: {
          size: 16,
          color: '#333'
        }
      },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      margin: {
        l: 20,
        r: 20,
        t: 60,
        b: 20
      },
      showlegend: true,
      legend: {
        orientation: 'v',
        x: 1.05,
        y: 0.5
      }
    }

    const config = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
    }

    Plotly.newPlot(containerId, [trace], layout, config)
  }

  /**
   * Draw a multi-line graph for comparing multiple datasets
   */
  static drawMultiLineGraph(
    containerId: string,
    data: Array<{
      x: number[],
      y: number[],
      name: string,
      color?: string
    }>,
    title: string = 'Multi-Line Graph',
    xLabel: string = 'X Axis',
    yLabel: string = 'Y Axis'
  ): void {
    const colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    const traces = data.map((dataset, index) => ({
      x: dataset.x,
      y: dataset.y,
      type: 'scatter',
      mode: 'lines+markers',
      name: dataset.name,
      line: {
        color: dataset.color || colors[index % colors.length],
        width: 2
      },
      marker: {
        size: 4,
        color: dataset.color || colors[index % colors.length]
      }
    }))

    const layout = {
      title: {
        text: title,
        font: {
          size: 16,
          color: '#333'
        }
      },
      xaxis: {
        title: {
          text: xLabel,
          font: {
            size: 12,
            color: '#666'
          }
        },
        gridcolor: '#f0f0f0',
        zeroline: false
      },
      yaxis: {
        title: {
          text: yLabel,
          font: {
            size: 12,
            color: '#666'
          }
        },
        gridcolor: '#f0f0f0',
        zeroline: false
      },
      plot_bgcolor: 'white',
      paper_bgcolor: 'white',
      margin: {
        l: 60,
        r: 40,
        t: 60,
        b: 60
      },
      showlegend: true,
      legend: {
        x: 0,
        y: 1,
        orientation: 'h'
      }
    }

    const config = {
      responsive: true,
      displayModeBar: true,
      displaylogo: false,
      modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d']
    }

    Plotly.newPlot(containerId, traces, layout, config)
  }

  /**
   * Update an existing line graph with new data
   */
  static updateLineGraph(
    containerId: string,
    xData: number[],
    yData: number[]
  ): void {
    const update = {
      x: [xData],
      y: [yData]
    }

    Plotly.restyle(containerId, update)
  }

  /**
   * Update an existing pie chart with new data
   */
  static updatePieChart(
    containerId: string,
    labels: string[],
    values: number[]
  ): void {
    const update = {
      labels: [labels],
      values: [values]
    }

    Plotly.restyle(containerId, update)
  }

  /**
   * Clear a graph container
   */
  static clearGraph(containerId: string): void {
    Plotly.purge(containerId)
  }

  /**
   * Resize a graph to fit its container
   */
  static resizeGraph(containerId: string): void {
    Plotly.Plots.resize(containerId)
  }

  /**
   * Export graph as PNG image
   */
  static exportAsPNG(containerId: string, filename: string = 'graph.png'): void {
    Plotly.downloadImage(containerId, {
      format: 'png',
      filename: filename,
      height: 600,
      width: 800
    })
  }

  /**
   * Export graph as SVG image
   */
  static exportAsSVG(containerId: string, filename: string = 'graph.svg'): void {
    Plotly.downloadImage(containerId, {
      format: 'svg',
      filename: filename,
      height: 600,
      width: 800
    })
  }
}
