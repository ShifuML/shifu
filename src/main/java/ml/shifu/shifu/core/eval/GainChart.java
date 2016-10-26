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

            writer.write("      </div>\n");
            writer.write("      <div class=\"col-sm-9 col-sm-offset-3 col-md-10 col-md-offset-2 main\">\n");
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container0"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container1"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container2"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container3"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container4"));
            writer.write(String.format(GainChartTemplate.HIGHCHART_DIV, "container5"));

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
                if(i != result.weightedGains.size() - 1) {
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
                if(i != result.weightedGains.size() - 1) {
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
                if(i != result.weightedGains.size() - 1) {
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
                if(i != result.weightedGains.size() - 1) {
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
            writer.write("  var ics = ['#container1','#container2', ,'#container4', ,'#container5'];\n");
            writer.write("  var icl = ics.length;\n");
            writer.write("  for (var i = 0; i < icl; i++) {\n");
            writer.write("      $(ics[i]).toggleClass('show');\n");
            writer.write("      $(ics[i]).toggleClass('hidden');\n");
            writer.write("      $(ics[i]).toggleClass('ls_chosen');\n");
            writer.write("  };\n");
            writer.write("\n");
            writer.write("});\n");
            writer.write("\n");
            writer.write("</script");
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
