import plotly.graph_objects as go

def create_gauge_chart(probability):
  #Determine color based on churn probability
  if probability < 0.3:
    color = "green"
  elif probability < 0.6:
    color = "yellow"
  else:
    color = "red"

  #Create a gauge chart
  fig = go.Figure(
    go.Indicator(mode="gauge+number",
                value=probability * 100,
                domain={
                  'x': [0,1],
                  'y': [0,1]
                },
                 number={'font': {
                   'size': 30,
                   'color': 'white'
                 },
                  'suffix': "%",
                  'valueformat': ".1f"
                },
                 gauge={
                   'axis': {
                     'range': [0, 100],
                     'tickwidth': 1,
                     'tickcolor': 'white',
                     'tickmode': 'array',
                     'tickvals': [0, 25, 50, 75, 100],  # Set tick values
                     'ticktext': ['0', '25', '50', '75', '100']  # Set tick labels
                   },
                   'bar': {
                     'color': color
                   },
                  'bgcolor': 'rgba(0,0,0,0)',
                   'borderwidth': 2,
                   'bordercolor': 'white',
                   'steps': [{
                     'range': [0, 30],
                     'color': 'rgba(0, 255, 0, 0.3)',
                   }, {
                     'range': [30, 60],
                     'color': 'rgba(255, 255, 0, 0.3)',
                   }, {
                     'range': [60, 100],
                     'color': 'rgba(255, 0, 0, 0.3)',
                   }],
                   'threshold': {
                     'line': {
                       'color': 'white',
                       'width': 4
                     },
                     'thickness': 0.75,
                     'value': 100,
                   }
                 }))

  #update chart layout
  fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': 'white'},
                    height=300,  # Set a fixed height
                    margin=dict(l=30, r=30, t=40, b=30))
  return fig

def create_model_probability_chart(probabilities):
  models = list(probabilities.keys())
  probs = list(probabilities.values())

  fig = go.Figure(data=[
    go.Bar(y=models,
           x=probs,
           orientation='h',
           text=[f'{p:.2%}' for p in probs],
           textposition='auto')
  ])

  fig.update_layout(
      yaxis_title='Models',
      xaxis_title='Probability',
      xaxis=dict(tickformat='.0%', range=[0, 1]),
      height=300,  # Reduced height to match gauge chart
      margin=dict(l=20, r=20, t=40, b=20),  # Adjusted margins
      paper_bgcolor="rgba(0,0,0,0)",
      plot_bgcolor="rgba(0,0,0,0)",
      font={'color': 'white'}
  )

  return fig

def create_percentile_chart(percentiles):
  x_values = [
      percentiles['NumOfProducts'],
      percentiles['Balance'],
      percentiles['EstimatedSalary'],
      percentiles['Tenure'],
      percentiles['CreditScore']
  ]

  fig = go.Figure(data=[
      go.Bar(
          y=['NumOfProducts', 'Balance', 'EstimatedSalary', 'Tenure', 'CreditScore'],
          x=x_values,
          orientation='h',
          textposition='auto'
      )
    ])

  fig.update_layout(
      title="Customer Percentiles",
      yaxis_title='Metric',
      xaxis_title='Percentile',
      xaxis=dict(
          tickvals=[0, 20, 40, 60, 80, 100],
          ticktext=['0%', '20%', '40%', '60%', '80%', '100%'],
          range=[0, 100]
      ),
      height=400,
      margin=dict(l=20, r=20, t=40, b=20)
  )

  return fig
