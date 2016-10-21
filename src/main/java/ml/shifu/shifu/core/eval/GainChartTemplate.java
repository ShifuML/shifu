/*
 * Copyright [2013-2016] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.eval;

/**
 * TODO
 *
 * @author Zhang David (pengzhang@paypal.com)
 */
public class GainChartTemplate {
    
    public static String[] DEFAULT_COLORS = new String[]{
            "#7cb5ec", "#f15c80", "#2b908f",
            "#f7a35c", "#8085e9", "#e4d354",
            "#90ed7d", "#f45b5b", "#91e8e1"};

    public static String[] DEFAULT_LINE_STYLES = new String[]{
            "Solid", "Dash", "DashDot",
            "ShortDot", "ShortDash", "ShortDashDot",
            "ShortDashDotDot", "Dot", "LongDash",
            "LongDashDot", "LongDashDotDot"};
            
    public static String HIGHCHART_BASE_BEGIN = new StringBuilder(1000)
            .append("<!DOCTYPE html>")
            .append("<html>")
            .append("<head>")
            .append("  <script src=\"http://cdn.bootcss.com/jquery/2.1.1/jquery.min.js\"></script>")
            .append("  <script src=\"http://cdn.bootcss.com/bootstrap/3.2.0/js/bootstrap.min.js\"></script>")
            .append("  <link href=\"http://cdn.bootcss.com/bootstrap/3.2.0/css/bootstrap.min.css\" rel=\"stylesheet\" />")
            .append("  <script src=\"http://cdn.bootcss.com/highcharts/4.0.4/highcharts.js\"></script>")
            .append("  <style type=\"text/css\">")
            .append("    .sidebar {")
            .append("      position: fixed;")
            .append("      top: 0px;")
            .append("      bottom: 0px;")
            .append("      left: 0px;")
            .append("      z-index: 1000;")
            .append("      display: block;")
            .append("      padding: 10px;")
            .append("      overflow-x: hidden;")
            .append("      overflow-y: auto;")
            .append("      background-color: #F5F5F5;")
            .append("      border-right: 1px solid #EEE;")
            .append("    }")
            .append("  </style>")
            .append("  <title>Advanced Gain Chart</title>")
            .append("</head>")
            .append("<body>")
            .append("  <div class=\"container-fluid\">")
            .append("    <div class=\"row\">")
            .append("      <div class=\"col-sm-3 col-md-2 sidebar\">%s</div>")
            .append("      <div class=\"col-sm-9 col-sm-offset-3 col-md-10 col-md-offset-2 main\">").toString();
         public static String HIGHCHART_SIDE_TEMPLATE = "%s\n%s\n";
         
         public static String HIGHCHART_BUTTON_PANEL_TEMPLATE = new StringBuilder(100)
            .append("      <div class=\"panel panel-info\">")
            .append("        <div class=\"panel-heading\">")
            .append("          <h3 class=\"panel-title\">%s</h3>")
            .append("        </div>")
            .append("        <div class=\"panel-body\">")
            .append("          %s")
            .append("        </div>")
            .append("      </div>").toString();
         
         public static String HIGHCHART_LIST_PANEL_TEMPLATE = new StringBuilder(100)
            .append("      <div class=\"panel panel-info\">")
            .append("        <div class=\"panel-heading\">")
            .append("          <h3 class=\"panel-title\">%s</h3>")
            .append("        </div>")
            .append("        <ul class=\"list-group\">")
            .append("          %s")
            .append("        </ul>")
            .append("      </div>").toString();
         
         public static String HIGHCHART_WGT_LIST_TEMPLATE = 
                 "<a id=\"%s\" href=\"#\" class=\"list-group-item%s\">%s</a>";
         
         public static String HIGHCHART_BUTTON_TEMPLATE = 
                 "<button id=\"%s\" type=\"button\" class=\"btn btn-success\">%s</button>";
         
         
         public static String HIGHCHART_DIV = 
                 "<div id=\"container%d\" sytle=\"height: 400px; min-width: 600px\" class=\"show\"></div>";

         public static String HIGHCHART_DATA_TEMPLATE = "\n<script>\n%s\n";
         
         public static String HIGHCHART_FUNCTION_TEMPLATE = new StringBuilder(100)
            .append("$(function () {")
            .append("  %s")
            .append("});")
            .append("")
            .append("$(document).ready(function() {")
            .append("  %s")
            .append("});")
            .append("</script>").toString();
            
         public static String HIGHCHART_SERIES_TEMPLATE = new StringBuilder(100)
              .append("      data: data_%d,")
              .append("      name: '%s',")
              .append("      dashStyle: '%s',")
              .append("      turboThreshold:0").toString();
         
         public static String HIGHCHART_TOOLTIP_TEMPLATE = "s += '%s: ' + this.point.%s + '%";
         
         public static String HIGHCHART_CHART_TEMPLATE = new StringBuilder(1000)
            .append("$('#container${cid}').highcharts({")
            .append("  chart: {")
            .append("    borderWidth: 1,")
            .append("    zoomType: 'x',")
            .append("    resetZoomButton: {")
            .append("      position: {")
            .append("        align: 'left',")
            .append("        x: 10,")
            .append("        y: 10")
            .append("      },")
            .append("      relativeTo: 'chart'")
            .append("    },")
            .append("    marginRight: 80")
            .append("  },")
            .append("  colors: [${colors}],")
            .append("  title: {")
            .append("    text : '${title}'")
            .append("  },")
            .append("  subtitle : {")
            .append("    text: '${subtitle}'")
            .append("  },")
            .append("  tooltip: {")
            .append("    shared: true,")
            .append("    useHTML: true,")
            .append("    headerFormat: '${score} <table> ${tiphead}',")
            .append("    pointFormat: ${tiptable},")
            .append("    footerFormat: '</table>',")
            .append("    crosshairs: true")
            .append("  },")
            .append("  yAxis: {")
            .append("    title: {")
            .append("      text: 'catch rate'")
            .append("    },")
            .append("    labels: {")
            .append("      formatter: function() {")
            .append("        return this.value + '%';")
            .append("      }")
            .append("    },")
            .append("    gridLineWidth: 1,")
            .append("    ceiling : 100,")
            .append("    floor : 0")
            .append("  },")
            .append("  xAxis: {")
            .append("    title: {")
            .append("      text: '${x}'")
            .append("    },")
            .append("    ${suf}")
            .append("    reversed: ${reverse},")
            .append("    gridLineWidth: 1")
            .append("  },")
            .append("  credits : {")
            .append("    enabled: false")
            .append("  },")
            .append("  legend: {")
            .append("    align : 'right',")
            .append("    verticalAlign: 'middle',")
            .append("    layout : 'vertical',")
            .append("    borderWidth : 1,")
            .append("    floating: true,")
            .append("    backgroundColor: 'white'")
            .append("  },")
            .append("  series: [${series}]")
            .append("});").toString();
         
         public static String HIGHCHART_GROUP_TOGGLE_TEMPLATE = new StringBuilder(200)
            .append("$('#%1$s').click(function() {")
            .append("  var cs = [%2$s];")
            .append("  var cl = cs.length;")
            .append("  for (var i = 0; i < cl; i++) {")
            .append("    if (! ($(cs[i]).hasClass('ls_chosen') || $(cs[i]).hasClass('bd_chosen'))) {")
            .append("      $(cs[i]).toggleClass('show');")
            .append("      $(cs[i]).toggleClass('hidden');")
            .append("    }")
            .append("    $(cs[i]).toggleClass('gp_chosen');")
            .append("  };")
            .append("  $('#%1$s').toggleClass('btn-success');")
            .append("  $('#%1$s').toggleClass('btn-default');")
            .append("});").toString();
         
         public static String HIGHCHART_BAD_TOGGLE_TEMPLATE = new StringBuilder(200)
            .append("$('#%1$s').click(function() {")
            .append("  var cs = [%2$s];")
            .append("  var cl = cs.length;")
            .append("  for (var i = 0; i < cl; i++) {")
            .append("    if (! ($(cs[i]).hasClass('ls_chosen') || $(cs[i]).hasClass('gp_chosen'))) {")
            .append("      $(cs[i]).toggleClass('show');")
            .append("      $(cs[i]).toggleClass('hidden');")
            .append("    }")
            .append("    $(cs[i]).toggleClass('bd_chosen');")
            .append("  };")
            .append("  $('#%1$s').toggleClass('btn-success');")
            .append("  $('#%1$s').toggleClass('btn-default');")
            .append("});").toString();
         
         public static String HIGHCHART_LIST_TOGGLE_TEMPLATE = new StringBuilder(200).append("")
            .append("$('#%1$s').click(function() {")
            .append("  var cs = [%2$s];")
            .append("  var cl = cs.length;")
            .append("  for (var i = 0; i < cl; i++) {")
            .append("    if (! ($(cs[i]).hasClass('gp_chosen') || $(cs[i]).hasClass('bd_chosen'))) {")
            .append("      $(cs[i]).toggleClass('show');")
            .append("      $(cs[i]).toggleClass('hidden');")
            .append("    }")
            .append("    $(cs[i]).toggleClass('ls_chosen');")
            .append("  };")
            .append("  $('#%1$s').toggleClass('list-group-item-success');")
            .append("});").toString();

         public static String HIGHCHART_READY_TOGGLE_TEMPLATE = new StringBuilder(1000).append("")
            .append("  var ics = [%s];")
            .append("  var icl = ics.length;")
            .append("  for (var i = 0; i < icl; i++) {")
            .append("    $(ics[i]).toggleClass('show');")
            .append("    $(ics[i]).toggleClass('hidden');")
            .append("    $(ics[i]).toggleClass('ls_chosen');")
            .append("  };").toString();
         
         public static String HIGHCHART_BASE_END = 
            "  </div>" +
            "</body>";
         
         public static String HIGHCHART_TIPTABLE_HEAD = "<tr><th>line</th><th>&nbsp;catch&nbsp;</th><th>&nbsp;fpr&nbsp;</th>";
         
         public static String HIGHCHART_TIPTABLE_STRING = "'<tr><td style=\"color:{series.color}\">{series.name}&nbsp;:</td><td>&nbsp;{point.y}%&nbsp;</td><td>,&nbsp;{point.fpr}&nbsp;</td>'";
         
}
