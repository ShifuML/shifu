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

import java.io.BufferedWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.List;

import ml.shifu.shifu.container.PerformanceObject;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.PerformanceResult;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.ShifuFileUtils;

/**
 * Generate gainchart with html format and csv format
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class GainChart {

    public static final DecimalFormat DF = new DecimalFormat("#.####");

    public void generateHtml(EvalConfig evalConfig, ModelConfig modelConfig, String fileName, PerformanceResult result)
            throws IOException {
        BufferedWriter writer = null;
        try {
            writer = ShifuFileUtils.getWriter(fileName, SourceType.LOCAL);

            writer.write(GainChartTemplate.HIGHCHART_BASE_BEGIN);

            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_1, "Weighted Operation Point",
                    "lst0", "Weighted Recall", "lst1", "Unit-wise Recall"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_2,
                    "Unit-wise Operation Point", "lst2", "Weighted Recall", "lst3", "Unit-wise Recall"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_3, "Model Score", "lst4",
                    "Weighted Recall", "lst5", "Unit-wise Recall"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_4, "Score Distibution",
                    "lst6", "Score Count"));

            writer.write("      </div>\n");
            writer.write("      <div class=\"col-sm-9 col-sm-offset-3 col-md-10 col-md-offset-2 main\">\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container0"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container1"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container2"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container3"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container4"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container5"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container6"));

            writer.write("<script>\n");
            writer.write("\n");
            writer.write("  var data_0 = [\n");
            for(int i = 0; i < result.weightedGains.size(); i++) {
                PerformanceObject po = result.weightedGains.get(i);
                writer.write(String.format(GainChartTemplate.DATA_FORMAT,
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.weightedPrecision * 100),
                        GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.weightedGains.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_1 = [\n");
            for(int i = 0; i < result.weightedGains.size(); i++) {
                PerformanceObject po = result.weightedGains.get(i);
                writer.write(String.format(GainChartTemplate.DATA_FORMAT, GainChartTemplate.DF.format(po.recall * 100),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.precision * 100),
                        GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.weightedGains.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_2 = [\n");
            for(int i = 0; i < result.gains.size(); i++) {
                PerformanceObject po = result.gains.get(i);
                writer.write(String.format(GainChartTemplate.DATA_FORMAT,
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.weightedPrecision * 100),
                        GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.gains.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_3 = [\n");
            for(int i = 0; i < result.gains.size(); i++) {
                PerformanceObject po = result.gains.get(i);
                writer.write(String.format(GainChartTemplate.DATA_FORMAT, GainChartTemplate.DF.format(po.recall * 100),
                        GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.precision * 100),
                        GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.gains.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_4 = [\n");
            for(int i = 0; i < result.modelScoreList.size(); i++) {
                PerformanceObject po = result.modelScoreList.get(i);
                writer.write(String.format(GainChartTemplate.DATA_FORMAT,
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.binLowestScore),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.weightedPrecision * 100),
                        GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.modelScoreList.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_5 = [\n");
            for(int i = 0; i < result.modelScoreList.size(); i++) {
                PerformanceObject po = result.modelScoreList.get(i);
                writer.write(String.format(GainChartTemplate.DATA_FORMAT, GainChartTemplate.DF.format(po.recall * 100),
                        GainChartTemplate.DF.format(po.binLowestScore),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.precision * 100),
                        GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.modelScoreList.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_6 = [\n");
            for(int i = 0; i < result.modelScoreList.size(); i++) {
                PerformanceObject po = result.modelScoreList.get(i);
                writer.write(String.format(GainChartTemplate.SCORE_DATA_FORMAT,
                        GainChartTemplate.DF.format(po.scoreCount), GainChartTemplate.DF.format(po.binLowestScore),
                        GainChartTemplate.DF.format(po.scoreCount), GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.modelScoreList.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("$(function () {\n");
            String fullName = modelConfig.getBasic().getName() + "::" + evalConfig.getName();
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE, "container0", "Weighted Recall",
                    modelConfig.getBasic().getName(), "Weighted  Operation Point", "%", "false", "data_0", "data_0",
                    fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE, "container1", "Unit-wise Recall",
                    modelConfig.getBasic().getName(), "Weighted  Operation Point", "%", "false", "data_1", "data_1",
                    fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE, "container2", "Weighted Recall",
                    modelConfig.getBasic().getName(), "Unit-wise  Operation Point", "%", "false", "data_2", "data_2",
                    fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE, "container3", "Unit-wise Recall",
                    modelConfig.getBasic().getName(), "Unit-wise  Operation Point", "%", "false", "data_3", "data_3",
                    fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE, "container4", "Weighted Recall",
                    modelConfig.getBasic().getName(), "Model Score", "", "true", "data_4", "data_4", fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE, "container5", "Unit-wise Recall",
                    modelConfig.getBasic().getName(), "Model Score", "", "true", "data_5", "data_5", fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.SCORE_HIGHCHART_CHART_TEMPLATE, "container6",
                    "Score Distribution", modelConfig.getBasic().getName(), "Model Score", "", "false", "data_6",
                    "data_6", fullName));
            writer.write("\n");
            writer.write("});\n");
            writer.write("\n");

            writer.write("$(document).ready(function() {\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst0", "container0", "lst0"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst1", "container1", "lst1"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst2", "container2", "lst2"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst3", "container3", "lst3"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst4", "container4", "lst4"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst5", "container5", "lst5"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst6", "container6", "lst6"));
            writer.write("\n");
            writer.write("  var ics = ['#container1','#container2', '#container4','#container5', '#container6'];\n");
            writer.write("  var icl = ics.length;\n");
            writer.write("  for (var i = 0; i < icl; i++) {\n");
            writer.write("      $(ics[i]).toggleClass('show');\n");
            writer.write("      $(ics[i]).toggleClass('hidden');\n");
            writer.write("      $(ics[i]).toggleClass('ls_chosen');\n");
            writer.write("  };\n");
            writer.write("\n");
            writer.write("});\n");
            writer.write("\n");
            writer.write("</script>\n");
            writer.write(GainChartTemplate.HIGHCHART_BASE_END);
        } finally {
            if(writer != null) {
                writer.close();
            }
        }
    }

    public void generateHtml4PrAndRoc(EvalConfig evalConfig, ModelConfig modelConfig, String fileName,
            PerformanceResult result) throws IOException {
        BufferedWriter writer = null;
        try {
            writer = ShifuFileUtils.getWriter(fileName, SourceType.LOCAL);

            writer.write(GainChartTemplate.HIGHCHART_BASE_BEGIN);

            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_1, "Weighted PR Curve",
                    "lst0", "Weighted Precision", "lst1", "Unit-wise Precision"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_2, "Unit-wise PR Curve",
                    "lst2", "Weighted Precision", "lst3", "Unit-wise Precision"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_1, "Weighted ROC Curve",
                    "lst4", "Weighted Recall", "lst5", "Unit-wise Recall"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_2, "Unit-wise ROC Curve",
                    "lst6", "Weighted Recall", "lst7", "Unit-wise Recall"));

            writer.write("      </div>\n");
            writer.write("      <div class=\"col-sm-9 col-sm-offset-3 col-md-10 col-md-offset-2 main\">\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container0"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container1"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container2"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container3"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container4"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container5"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container6"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container7"));

            writer.write("<script>\n");
            writer.write("\n");
            writer.write("  var data_0 = [\n");

            for(int i = 0; i < result.weightedPr.size(); i++) {
                PerformanceObject po = result.weightedPr.get(i);
                writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                        GainChartTemplate.DF.format(po.weightedPrecision * 100),
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.weightedPrecision * 100),
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.weightedFpr * 100),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.weightedPr.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_1 = [\n");
            for(int i = 0; i < result.weightedPr.size(); i++) {
                PerformanceObject po = result.weightedPr.get(i);
                writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                        GainChartTemplate.DF.format(po.precision * 100),
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.precision * 100),
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.weightedFpr * 100),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.weightedPr.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_2 = [\n");
            for(int i = 0; i < result.pr.size(); i++) {
                PerformanceObject po = result.pr.get(i);
                writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                        GainChartTemplate.DF.format(po.weightedPrecision * 100),
                        GainChartTemplate.DF.format(po.recall * 100),
                        GainChartTemplate.DF.format(po.weightedPrecision * 100),
                        GainChartTemplate.DF.format(po.recall * 100), GainChartTemplate.DF.format(po.fpr * 100),
                        GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.pr.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_3 = [\n");
            for(int i = 0; i < result.pr.size(); i++) {
                PerformanceObject po = result.pr.get(i);
                writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                        GainChartTemplate.DF.format(po.precision * 100), GainChartTemplate.DF.format(po.recall * 100),
                        GainChartTemplate.DF.format(po.precision * 100), GainChartTemplate.DF.format(po.recall * 100),
                        GainChartTemplate.DF.format(po.fpr * 100), GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.pr.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_4 = [\n");
            for(int i = 0; i < result.weightedRoc.size(); i++) {
                PerformanceObject po = result.weightedRoc.get(i);
                writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.weightedFpr * 100),
                        GainChartTemplate.DF.format(po.weightedPrecision * 100),
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.weightedFpr * 100),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.weightedRoc.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_5 = [\n");
            for(int i = 0; i < result.weightedRoc.size(); i++) {
                PerformanceObject po = result.weightedRoc.get(i);
                writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                        GainChartTemplate.DF.format(po.recall * 100),
                        GainChartTemplate.DF.format(po.weightedFpr * 100),
                        GainChartTemplate.DF.format(po.weightedPrecision * 100),
                        GainChartTemplate.DF.format(po.recall * 100),
                        GainChartTemplate.DF.format(po.weightedFpr * 100),
                        GainChartTemplate.DF.format(po.weightedActionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.weightedRoc.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_6 = [\n");
            for(int i = 0; i < result.roc.size(); i++) {
                PerformanceObject po = result.roc.get(i);
                writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.fpr * 100), GainChartTemplate.DF.format(po.precision * 100),
                        GainChartTemplate.DF.format(po.weightedRecall * 100),
                        GainChartTemplate.DF.format(po.fpr * 100), GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.roc.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("  var data_7 = [\n");
            for(int i = 0; i < result.roc.size(); i++) {
                PerformanceObject po = result.roc.get(i);
                writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                        GainChartTemplate.DF.format(po.recall * 100), GainChartTemplate.DF.format(po.fpr * 100),
                        GainChartTemplate.DF.format(po.precision * 100), GainChartTemplate.DF.format(po.recall * 100),
                        GainChartTemplate.DF.format(po.fpr * 100), GainChartTemplate.DF.format(po.actionRate * 100),
                        GainChartTemplate.DF.format(po.binLowestScore)));
                if(i != result.roc.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("  ];\n");
            writer.write("\n");

            writer.write("$(function () {\n");
            String fullName = modelConfig.getBasic().getName() + "::" + evalConfig.getName();
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE2, "container0",
                    "Weighted Recall - Weighted Precision (PR Curve)", modelConfig.getBasic().getName(),
                    "Weighte Precision", "Weighted Recall", "%", "false", "data_0", "data_0", fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE2, "container1",
                    "Weighted Recall - Unit-wise Precision (PR Curve)", modelConfig.getBasic().getName(),
                    "Unit-wise Precision", "Weighted Recall", "%", "false", "data_1", "data_1", fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE2, "container2",
                    "Unit-wise Recall - Weighted Precision (PR Curve)", modelConfig.getBasic().getName(),
                    "Weighted Precision", "Unit-wise Recall", "%", "false", "data_2", "data_2", fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE2, "container3",
                    "Unit-wise Recall - Unit-wise Precision (PR Curve)", modelConfig.getBasic().getName(),
                    "Unit-wise  Precision", "Unit-wise Recall", "%", "false", "data_3", "data_3", fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE2, "container4",
                    "Weighted False Positive Rate - Weighted Recall (ROC Curve)", modelConfig.getBasic().getName(),
                    "Weighted Recall", "Weighted False Positive Rate", "%", "false", "data_4", "data_4", fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE2, "container5",
                    "Weighted False Positive Rate  - Unit-wise Recall (ROC Curve)", modelConfig.getBasic().getName(),
                    "Unit-wise Recall", "Weighted False Positive Rate", "%", "false", "data_5", "data_5", fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE2, "container6",
                    "Unit-wise False Positive Rate  - Weighted Recall (ROC Curve)", modelConfig.getBasic().getName(),
                    "Weighted Recall", "Unit-wise False Positive Rate", "%", "false", "data_6", "data_6", fullName));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE2, "container7",
                    "Unit-wise False Positive Rate  - Unit-wise Recall (ROC Curve)", modelConfig.getBasic().getName(),
                    "Unit-wise Recall", "Unit-wise False Positive Rate", "%", "false", "data_7", "data_7", fullName));
            writer.write("\n");
            writer.write("});\n");
            writer.write("\n");

            writer.write("$(document).ready(function() {\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst0", "container0", "lst0"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst1", "container1", "lst1"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst2", "container2", "lst2"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst3", "container3", "lst3"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst4", "container4", "lst4"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst5", "container5", "lst5"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst6", "container6", "lst6"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst7", "container7", "lst7"));
            writer.write("\n");
            writer.write("  var ics = ['#container1','#container2','#container5','#container6'];\n");
            writer.write("  var icl = ics.length;\n");
            writer.write("  for (var i = 0; i < icl; i++) {\n");
            writer.write("      $(ics[i]).toggleClass('show');\n");
            writer.write("      $(ics[i]).toggleClass('hidden');\n");
            writer.write("      $(ics[i]).toggleClass('ls_chosen');\n");
            writer.write("  };\n");
            writer.write("\n");
            writer.write("});\n");
            writer.write("\n");
            writer.write("</script>\n");
            writer.write(GainChartTemplate.HIGHCHART_BASE_END);
        } finally {
            if(writer != null) {
                writer.close();
            }
        }
    }

    public void generateHtml(EvalConfig evalConfig, ModelConfig modelConfig, String fileName,
            List<PerformanceResult> results, List<String> names) throws IOException {
        BufferedWriter writer = null;
        try {
            writer = ShifuFileUtils.getWriter(fileName, SourceType.LOCAL);

            writer.write(GainChartTemplate.HIGHCHART_BASE_BEGIN);

            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_1, "Weighted Operation Point",
                    "lst0", "Weighted Recall", "lst1", "Unit-wise Recall"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_2,
                    "Unit-wise Operation Point", "lst2", "Weighted Recall", "lst3", "Unit-wise Recall"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_3, "Model Score", "lst4",
                    "Weighted Recall", "lst5", "Unit-wise Recall"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_4, "Score Distibution",
                    "lst6", "Score Count"));

            writer.write("      </div>\n");
            writer.write("      <div class=\"col-sm-9 col-sm-offset-3 col-md-10 col-md-offset-2 main\">\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container0"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container1"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container2"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container3"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container4"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container5"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container6"));

            writer.write("<script>\n");
            writer.write("\n");

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + j + " = [\n");
                for(int i = 0; i < result.weightedGains.size(); i++) {
                    PerformanceObject po = result.weightedGains.get(i);
                    writer.write(String.format(GainChartTemplate.DATA_FORMAT,
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.weightedPrecision * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.weightedGains.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (results.size() + j) + " = [\n");
                for(int i = 0; i < result.weightedGains.size(); i++) {
                    PerformanceObject po = result.weightedGains.get(i);
                    writer.write(String.format(GainChartTemplate.DATA_FORMAT,
                            GainChartTemplate.DF.format(po.recall * 100),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.precision * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.weightedGains.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (2 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.gains.size(); i++) {
                    PerformanceObject po = result.gains.get(i);
                    writer.write(String.format(GainChartTemplate.DATA_FORMAT,
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.weightedPrecision * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.gains.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (3 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.gains.size(); i++) {
                    PerformanceObject po = result.gains.get(i);
                    writer.write(String.format(GainChartTemplate.DATA_FORMAT,
                            GainChartTemplate.DF.format(po.recall * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.precision * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.gains.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (4 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.modelScoreList.size(); i++) {
                    PerformanceObject po = result.modelScoreList.get(i);
                    writer.write(String.format(GainChartTemplate.DATA_FORMAT,
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.binLowestScore),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.weightedPrecision * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.modelScoreList.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (5 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.modelScoreList.size(); i++) {
                    PerformanceObject po = result.modelScoreList.get(i);
                    writer.write(String.format(GainChartTemplate.DATA_FORMAT,
                            GainChartTemplate.DF.format(po.recall * 100),
                            GainChartTemplate.DF.format(po.binLowestScore),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.precision * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.modelScoreList.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (6 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.modelScoreList.size(); i++) {
                    PerformanceObject po = result.modelScoreList.get(i);
                    writer.write(String.format(GainChartTemplate.SCORE_DATA_FORMAT,
                            GainChartTemplate.DF.format(po.scoreCount), GainChartTemplate.DF.format(po.binLowestScore),
                            GainChartTemplate.DF.format(po.scoreCount), GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.modelScoreList.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            writer.write("$(function () {\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX, "container0",
                    "Weighted Recall", modelConfig.getBasic().getName(), "Weighted  Operation Point", "%", "false"));
            int currIndex = 0;
            writer.write("series: [");
            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX, "container1",
                    "Unit-wise Recall", modelConfig.getBasic().getName(), "Weighted  Operation Point", "%", "false"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX, "container2",
                    "Weighted Recall", modelConfig.getBasic().getName(), "Unit-wise  Operation Point", "%", "false"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX, "container3",
                    "Unit-wise Recall", modelConfig.getBasic().getName(), "Unit-wise  Operation Point", "%", "false"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX, "container4",
                    "Weighted Recall", modelConfig.getBasic().getName(), "Model Score", "", "true"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX, "container5",
                    "Unit-wise Recall", modelConfig.getBasic().getName(), "Model Score", "", "true"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.SCORE_HIGHCHART_CHART_PREFIX, "container6",
                    "Score Distribution", modelConfig.getBasic().getName(), "Model Score", "", "false"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write("});\n");
            writer.write("\n");

            writer.write("$(document).ready(function() {\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst0", "container0", "lst0"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst1", "container1", "lst1"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst2", "container2", "lst2"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst3", "container3", "lst3"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst4", "container4", "lst4"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst5", "container5", "lst5"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst6", "container6", "lst6"));
            writer.write("\n");
            writer.write("  var ics = ['#container1', '#container2', '#container4', '#container5', '#container6'];\n");
            writer.write("  var icl = ics.length;\n");
            writer.write("  for (var i = 0; i < icl; i++) {\n");
            writer.write("      $(ics[i]).toggleClass('show');\n");
            writer.write("      $(ics[i]).toggleClass('hidden');\n");
            writer.write("      $(ics[i]).toggleClass('ls_chosen');\n");
            writer.write("  };\n");
            writer.write("\n");
            writer.write("});\n");
            writer.write("\n");
            writer.write("</script>\n");
            writer.write(GainChartTemplate.HIGHCHART_BASE_END);
        } finally {
            if(writer != null) {
                writer.close();
            }
        }
    }

    public void generateHtml4PrAndRoc(EvalConfig evalConfig, ModelConfig modelConfig, String fileName,
            List<PerformanceResult> results, List<String> names) throws IOException {
        BufferedWriter writer = null;
        try {
            writer = ShifuFileUtils.getWriter(fileName, SourceType.LOCAL);

            writer.write(GainChartTemplate.HIGHCHART_BASE_BEGIN);

            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_1, "Weighted PR Curve",
                    "lst0", "Weighted Precision", "lst1", "Unit-wise Precision"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_2, "Unit-wise PR Curve",
                    "lst2", "Weighted Precision", "lst3", "Unit-wise Precision"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_1, "Weighted ROC Curve",
                    "lst4", "Weighted Recall", "lst5", "Unit-wise Recall"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_BUTTON_PANEL_TEMPLATE_2, "Unit-wise ROC Curve",
                    "lst6", "Weighted Recall", "lst7", "Unit-wise Recall"));

            writer.write("      </div>\n");
            writer.write("      <div class=\"col-sm-9 col-sm-offset-3 col-md-10 col-md-offset-2 main\">\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container0"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container1"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container2"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container3"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container4"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container5"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container6"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container7"));

            writer.write("<script>\n");
            writer.write("\n");

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + j + " = [\n");
                for(int i = 0; i < result.weightedPr.size(); i++) {
                    PerformanceObject po = result.weightedPr.get(i);
                    writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                            GainChartTemplate.DF.format(po.weightedPrecision * 100),
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.weightedPrecision * 100),
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.weightedFpr * 100),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.weightedPr.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (results.size() + j) + " = [\n");
                for(int i = 0; i < result.weightedPr.size(); i++) {
                    PerformanceObject po = result.weightedPr.get(i);
                    writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                            GainChartTemplate.DF.format(po.precision * 100),
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.precision * 100),
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.weightedFpr * 100),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.weightedPr.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (2 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.pr.size(); i++) {
                    PerformanceObject po = result.pr.get(i);
                    writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                            GainChartTemplate.DF.format(po.weightedPrecision * 100),
                            GainChartTemplate.DF.format(po.recall * 100),
                            GainChartTemplate.DF.format(po.weightedPrecision * 100),
                            GainChartTemplate.DF.format(po.recall * 100), GainChartTemplate.DF.format(po.fpr * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.pr.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (3 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.pr.size(); i++) {
                    PerformanceObject po = result.pr.get(i);
                    writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                            GainChartTemplate.DF.format(po.precision * 100),
                            GainChartTemplate.DF.format(po.recall * 100),
                            GainChartTemplate.DF.format(po.precision * 100),
                            GainChartTemplate.DF.format(po.recall * 100), GainChartTemplate.DF.format(po.fpr * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.pr.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (4 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.weightedRoc.size(); i++) {
                    PerformanceObject po = result.weightedRoc.get(i);
                    writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.weightedFpr * 100),
                            GainChartTemplate.DF.format(po.weightedPrecision * 100),
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.weightedFpr * 100),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.weightedRoc.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (5 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.weightedRoc.size(); i++) {
                    PerformanceObject po = result.weightedRoc.get(i);
                    writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                            GainChartTemplate.DF.format(po.recall * 100),
                            GainChartTemplate.DF.format(po.weightedFpr * 100),
                            GainChartTemplate.DF.format(po.weightedPrecision * 100),
                            GainChartTemplate.DF.format(po.recall * 100),
                            GainChartTemplate.DF.format(po.weightedFpr * 100),
                            GainChartTemplate.DF.format(po.weightedActionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.weightedRoc.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (6 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.roc.size(); i++) {
                    PerformanceObject po = result.roc.get(i);
                    writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.fpr * 100), GainChartTemplate.DF.format(po.precision * 100),
                            GainChartTemplate.DF.format(po.weightedRecall * 100),
                            GainChartTemplate.DF.format(po.fpr * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.roc.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            for(int j = 0; j < results.size(); j++) {
                PerformanceResult result = results.get(j);
                writer.write("  var data_" + (7 * results.size() + j) + " = [\n");
                for(int i = 0; i < result.roc.size(); i++) {
                    PerformanceObject po = result.roc.get(i);
                    writer.write(String.format(GainChartTemplate.PRROC_DATA_FORMAT,
                            GainChartTemplate.DF.format(po.recall * 100), GainChartTemplate.DF.format(po.fpr * 100),
                            GainChartTemplate.DF.format(po.precision * 100),
                            GainChartTemplate.DF.format(po.recall * 100), GainChartTemplate.DF.format(po.fpr * 100),
                            GainChartTemplate.DF.format(po.actionRate * 100),
                            GainChartTemplate.DF.format(po.binLowestScore)));
                    if(i != result.roc.size() - 1) {
                        writer.write(",");
                    }
                }
                writer.write("  ];\n");
                writer.write("\n");
            }

            writer.write("$(function () {\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX3, "container0",
                    "Weighted Recall - Weighted Precision (PR Curve)", modelConfig.getBasic().getName(),
                    "Weighte Precision", "Weighted Recall", "%", "false"));

            int currIndex = 0;
            writer.write("series: [");
            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX3, "container1",
                    "Weighted Recall - Unit-wise Precision (PR Curve)", modelConfig.getBasic().getName(),
                    "Unit-wise Precision", "Weighted Recall", "%", "false"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX3, "container2",
                    "Unit-wise Recall - Weighted Precision (PR Curve)", modelConfig.getBasic().getName(),
                    "Weighted Precision", "Unit-wise Recall", "%", "false"));

            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX3, "container3",
                    "Unit-wise Recall - Unit-wise Precision (PR Curve)", modelConfig.getBasic().getName(),
                    "Unit-wise Precision", "Unit-wise Recall", "%", "false"));

            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX3, "container4",
                    "Weighted FPR - Weighted Recall (ROC Curve)", modelConfig.getBasic().getName(), "Weighted Recall",
                    "Weighted FPR", "%", "false"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX3, "container5",
                    "Weighted FPR - Unit-wise Recall (ROC Curve)", modelConfig.getBasic().getName(),
                    "Unit-wise Recall", "Weighted FPR", "%", "false"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX3, "container6",
                    "Unit-wise FPR - Weighted Recall (ROC Curve)", modelConfig.getBasic().getName(), "Weighted Recall",
                    "Unit-wise FPR", "%", "false"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write(String.format(GainChartTemplate.HIGHCHART_CHART_TEMPLATE_PREFIX3, "container7",
                    "Unit-wise FPR - Unit-wise Recall (ROC Curve)", modelConfig.getBasic().getName(),
                    "Unit-wise Recall", "Unit-wise FPR", "%", "false"));
            writer.write("series: [");

            for(int i = 0; i < results.size(); i++) {
                writer.write("{");
                writer.write("  data: data_" + (currIndex++) + ",");
                writer.write("  name: '" + names.get(i) + "',");
                writer.write("  turboThreshold:0");
                writer.write("}");
                if(i != results.size() - 1) {
                    writer.write(",");
                }
            }
            writer.write("]");
            writer.write("});");
            writer.write("\n");

            writer.write("});\n");
            writer.write("\n");

            writer.write("$(document).ready(function() {\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst0", "container0", "lst0"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst1", "container1", "lst1"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst2", "container2", "lst2"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst3", "container3", "lst3"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst4", "container4", "lst4"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst5", "container5", "lst5"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst6", "container6", "lst6"));
            writer.write("\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_LIST_TOGGLE_TEMPLATE, "lst7", "container7", "lst7"));
            writer.write("\n");
            writer.write("\n");
            writer.write("  var ics = ['#container1','#container2', '#container5','#container6'];\n");
            writer.write("  var icl = ics.length;\n");
            writer.write("  for (var i = 0; i < icl; i++) {\n");
            writer.write("      $(ics[i]).toggleClass('show');\n");
            writer.write("      $(ics[i]).toggleClass('hidden');\n");
            writer.write("      $(ics[i]).toggleClass('ls_chosen');\n");
            writer.write("  };\n");
            writer.write("\n");
            writer.write("});\n");
            writer.write("\n");
            writer.write("</script>\n");
            writer.write(GainChartTemplate.HIGHCHART_BASE_END);
        } finally {
            if(writer != null) {
                writer.close();
            }
        }
    }

    public void generateCsv(EvalConfig evalConfig, ModelConfig modelConfig, String fileName,
            List<PerformanceObject> performanceList) throws IOException {
        BufferedWriter writer = null;
        try {
            writer = ShifuFileUtils.getWriter(fileName, SourceType.LOCAL);
            writer.write("ActionRate,WeightedActionRate,Recall,WeightedRecall,Precision,WeightedPrecision,FPR,WeightedFPR,BinLowestScore\n");
            String formatString = "%s,%s,%s,%s,%s,%s,%s,%s,%s\n";

            for(PerformanceObject po: performanceList) {
                writer.write(String.format(formatString, DF.format(po.actionRate), DF.format(po.weightedActionRate),
                        DF.format(po.recall), DF.format(po.weightedRecall), DF.format(po.precision),
                        DF.format(po.weightedPrecision), DF.format(po.fpr), DF.format(po.weightedFpr),
                        po.binLowestScore));
            }
        } finally {
            if(writer != null) {
                writer.close();
            }
        }
    }

}
