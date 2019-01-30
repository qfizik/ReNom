{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. auto{{objtype}}:: {{ objname }}

    {% block methods %}

    {% if methods %}
    .. rubric:: Method Summary

    .. autosummary::
 
    {% for item in methods %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Attributes

    .. autosummary::
    {% for item in attributes %}
        ~{{ name }}.{{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% for item in methods %}
    .. automethod:: {{ item }}
    {%- endfor %}

    {% endblock %}

