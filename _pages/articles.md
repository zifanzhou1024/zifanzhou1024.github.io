---
layout: archive
permalink: /articles/
title: "Articles"
author_profile: true
---

{% assign sorted_articles = site.articles | sort: "date" | reverse %}

{% for post in sorted_articles %}
  {% include archive-single.html %}
{% endfor %}
