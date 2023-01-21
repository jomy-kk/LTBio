{% import 'macros.rst' as macros %}

{% if not obj.display %}
:orphan:

{% endif %}
:py:mod:`{{ obj.name }}`
=========={{ "=" * obj.name|length }}

.. py:module:: {{ obj.name }}

{% if obj.docstring %}
.. autoapi-nested-parse::

   {{ obj.docstring|indent(3) }}

{% endif %}

Overview
{{ "-" * obj.type|length }}---------

{% block subpackages %}
{% set visible_subpackages = obj.subpackages|selectattr("display")|list %}
{% if visible_subpackages %}
Subpackages
~~~~~~~
.. toctree::
   :titlesonly:
   :maxdepth: 1

{% for subpackage in visible_subpackages %}
   {{ subpackage.short_name }}/index.rst
{% endfor %}


{% endif %}
{% endblock %}

{% block submodules %}
{% set visible_submodules = obj.submodules|selectattr("display")|list %}
{% if visible_submodules %}
Submodules
~~~~~~~
.. toctree::
   :titlesonly:
   :maxdepth: 1

{% for submodule in visible_submodules %}
   {{ submodule.short_name }}/index.rst
{% endfor %}


{% endif %}
{% endblock %}
{% block content %}
{% if obj.all is not none %}
{% set visible_children = obj.children|selectattr("short_name", "in", obj.all)|list %}
{% elif obj.type is equalto("package") %}
{% set visible_children = obj.children|selectattr("display")|list %}
{% else %}
{% set visible_children = obj.children|selectattr("display")|rejectattr("imported")|list %}
{% endif %}
{% if visible_children %}


{% set visible_classes = visible_children|selectattr("type", "equalto", "class")|list %}
{% set visible_functions = visible_children|selectattr("type", "equalto", "function")|list %}
{% set visible_attributes = visible_children|selectattr("type", "equalto", "data")|list %}
{% if "show-module-summary" in autoapi_options and (visible_classes or visible_functions) %}


{% block classes scoped %}
{% if visible_classes %}
Classes
~~~~~~~
{{ macros.auto_summary(visible_classes, title="Classes") }}
{% endif %}
{% endblock %}



{% block functions scoped %}
{% if visible_functions %}
Functions
~~~~~~~~~
{{ macros.auto_summary(visible_functions, title="Functions") }}
{% endif %}
{% endblock %}





Contents
{{ "-" * obj.type|length }}---------

{% block attributes scoped %}
{% if visible_attributes %}
Attributes
~~~~~~~~~~

.. autoapisummary::

{% for attribute in visible_attributes %}
   {{ attribute.id }}
{% endfor %}


{% endif %}
{% endblock %}
{% endif %}
{% for obj_item in visible_children %}
{{ obj_item.render()|indent(0) }}
{% endfor %}
{% endif %}
{% endblock %}
