from graph_element import graph_element, operational_element

class learnable_graph_element(graph_element):
  '''
    A learnable graph element is responsible for storing and performing the forward, backward and update operations in a normal neural-network setting.
  '''
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs) 
    
    

    
