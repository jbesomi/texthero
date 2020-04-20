import plotly.express as px

def scatterplot(df, column, color, hover_data):
    pca0 = df[column].apply(lambda x: x[0])
    pca1 = df[column].apply(lambda x: x[1])

    fig = px.scatter(df,
                     x=pca0,
                     y=pca1,
                     color=color,
                     hover_data=hover_data)

    fig.show(config={'displayModeBar': False})
