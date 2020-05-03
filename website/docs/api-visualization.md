---
id: api-visualization 
title: Visualization
---

# Visualization

Visualize insights and statistics of a text-based Pandas DataFrame.


### texthero.visualization.scatterplot(df, col, color=None, hover_data=None, title='', return_figure=False)
Show scatterplot using python plotly scatter.


* **Parameters**

    
    * **df** – 


    * **col** – The name of the column of the DataFrame used for x and y axis.



### texthero.visualization.top_words(s, normalize=False)
Return most common words.


* **Parameters**

    
    * **s** (`Series`) – 


    * **normalize** – Default is False. If set to True, returns normalized values.



* **Return type**

    `Series`
