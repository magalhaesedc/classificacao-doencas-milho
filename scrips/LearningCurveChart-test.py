import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.io as pio

df_train_EX01EM = pd.read_csv('accuracy_test_EX01EM.csv', names=['train'])
df_train_EX05LM = pd.read_csv('accuracy_train_EX05LM.csv', names=['train'])
#df_train_EX02EM = pd.read_csv('accuracy_test_EX02EM.csv', names=['train'])
df_train_EX03EM = pd.read_csv('accuracy_test_EX03EM.csv', names=['train'])
#df_train_EX04EM = pd.read_csv('accuracy_test_EX04EM.csv', names=['train'])
df_train_EX01DO = pd.read_csv('accuracy_test_EX01DO.csv')

df_test_EX01EM = pd.read_csv('accuracy_train_EX01EM.csv', names=['test'])
df_test_EX05LM = pd.read_csv('accuracy_test_EX05LM.csv', names=['test'])
#df_test_EX02EM = pd.read_csv('accuracy_train_EX02EM.csv', names=['test'])
df_test_EX03EM = pd.read_csv('accuracy_train_EX03EM.csv', names=['test'])
#df_test_EX04EM = pd.read_csv('accuracy_train_EX04EM.csv', names=['test'])
df_test_EX01DO = pd.read_csv('accuracy_train_EX01DO.csv')

y_train_EX01DO = df_train_EX01DO.loc[5,]
y_test_EX01DO = df_test_EX01DO.loc[5,]
y_train_EX01DO = y_train_EX01DO[1:]
y_test_EX01DO = y_test_EX01DO[1:]

y_train_EX01EM = df_train_EX01EM["train"]
y_test_EX01EM = df_test_EX01EM["test"]

y_train_EX05LM = df_train_EX05LM["train"]
y_test_EX05LM = df_test_EX05LM["test"]

#y_train_EX02EM = df_train_EX02EM["train"]
#y_test_EX02EM = df_test_EX02EM["test"]

y_train_EX03EM = df_train_EX03EM["train"]
y_test_EX03EM = df_test_EX03EM["test"]

#y_train_EX04EM = df_train_EX04EM["train"]
#y_test_EX04EM = df_test_EX04EM["test"]

X30 = np.arange(1, 31)
X100 = np.arange(1, 101)

#TESTE
fig = go.Figure([
    go.Scatter(
        name='Cenário 1',
        x=X30,
        y=y_test_EX01EM,
        #mode='markers+lines',
        marker=dict(color='#0000CD', size=2),
        #marker=dict(color='red', size=7),
        showlegend=True
    ),
    go.Scatter(
        name='Cenário 2',
        x=X30,
        y=y_train_EX05LM,
        #mode='markers',
        marker=dict(color='#32CD32', size=2),
        showlegend=True
    ),
    go.Scatter(
        name='Cenário 3',
        x=X30,
        y=y_test_EX03EM,
        #mode='markers',
        marker=dict(color='#FF0000', size=2),
        showlegend=True
    ),
    go.Scatter(
        name='Cenário 4',
        x=X30,
        y=y_test_EX01DO,
        #mode='markers',
        marker=dict(color='#FFD700', size=2),
        showlegend=True
    )
])
#Metodo add_vline é suportado somente nas versões >4.12 do Plotly
#fig.add_vline(x='2021-02-10', line_width=2, line_dash="dash", line_color="DimGray")
fig.update_layout(
    #yaxis_title='Wind speed (m/s)',
    title='Learning curve - Test',
    hovermode="x")
#plotly.offline.plot(fig)
pio.write_image(fig, "learning_curve-test.svg")
fig.show()