{{ fullname | escape | underline }}

Description
-----------

.. automodule:: {{ fullname }}

.. currentmodule:: {{ fullname }}

{% if classes %}
.. rubric:: Classes

.. autosummary::
    :toctree: {{ name }}
    {% for class in classes %}
    {{ class }}
    {% endfor %}

{% endif %}

{% if functions %}
.. rubric:: Functions

.. autosummary::
    :toctree: {{ name }}
    {% for function in functions %}
    {{ function }}
    {% endfor %}

{% endif %}
