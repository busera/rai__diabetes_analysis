{%- extends 'html/index.html.j2' -%}

{% block input %}
<details>
  <summary style="cursor:pointer; font-weight: bold; background: #f0f0f0; padding: 5px; border: 1px solid #ddd;">
    Click to show/hide code
  </summary>
  <div class="cell code_cell">
    {{ super() }}
  </div>
</details>
{% endblock input %}