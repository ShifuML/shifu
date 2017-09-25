/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;

import javax.xml.transform.sax.SAXSource;

import ml.shifu.guagua.util.FileUtils;
import ml.shifu.shifu.container.ScoreObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.Scorer;
import ml.shifu.shifu.core.dtrain.dataset.BasicFloatNetwork;
import ml.shifu.shifu.core.dtrain.nn.IndependentNNModel;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.HDFSUtils;

import org.apache.hadoop.fs.Path;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.encog.ml.BasicML;
import org.jpmml.evaluator.EvaluatorUtil;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.MiningModelEvaluator;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;
import org.xml.sax.InputSource;

public class NNModelEvalAndScoreTest {

    private IndependentNNModel iNNModel;
    private BasicFloatNetwork nnModel;
    private Scorer scorer;
    private MiningModelEvaluator evaluator;

    @BeforeClass
    public void setUp() throws IOException {
        String modelPath = "src/test/resources/dttest/model/nn/binary.nn";
        FileInputStream fi = null;
        try {
            fi = new FileInputStream(modelPath);
            iNNModel = IndependentNNModel.loadFromStream(fi);
        } finally {
            fi.close();
        }

        ModelConfig modelConfig = CommonUtils.loadModelConfig("src/test/resources/dttest/config/nn/ModelConfig.json",
                SourceType.LOCAL);
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/dttest/config/nn/ColumnConfig.json", SourceType.LOCAL);

        nnModel = (BasicFloatNetwork) CommonUtils.loadModel(modelConfig, new Path(
                "src/test/resources/dttest/model/nn/encog.nn"), HDFSUtils.getLocalFS());

        List<BasicML> models = new ArrayList<BasicML>();
        models.add(nnModel);
        scorer = new Scorer(models, columnConfigList, "NN", modelConfig);

        evaluator = new MiningModelEvaluator(loadPMML(new File("src/test/resources/dttest/model/example.pmml")));
    }

    private static PMML loadPMML(File file) {
        InputStream is = null;
        try {
            is = new FileInputStream(file);
            InputSource source = new InputSource(is);
            SAXSource transformedSource = ImportFilter.apply(source);
            return JAXBUtil.unmarshalPMML(transformedSource);
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            if(is != null) {
                try {
                    is.close();
                } catch (IOException ignore) {
                }
            }
        }
    }

    @SuppressWarnings("unchecked")
    @Test
    public void testPMML() throws IOException {
        List<FieldName> activeFields = this.evaluator.getActiveFields();
        HashSet<String> activeNames = new HashSet<String>();
        for(FieldName fn: activeFields) {
            activeNames.add(fn.getValue());
        }

        List<String> lines = FileUtils.readLines(new File("src/test/resources/dttest/data/example.csv"));

        if(lines.size() <= 1) {
            return;
        }
        String[] headers = CommonUtils.split(lines.get(0), "|");
        // score with format <String, String>
        for(int i = 1; i < lines.size(); i++) {
            Map<FieldName, FieldValue> dataMap = new HashMap<FieldName, FieldValue>();

            Map<String, String> map = new HashMap<String, String>();

            String[] data = CommonUtils.split(lines.get(i), "|");;
            // System.out.println("data len is " + data.length);
            if(data.length != headers.length) {
                System.out.println("One invalid input data");
                break;
            }

            for(int j = 0; j < headers.length; j++) {
                map.put(headers[j], data[j]);
            }

            for(FieldName fn: activeFields) {
                String valueStr = map.get(fn.getValue());
                if(("").equals(valueStr) || ("NA").equals(valueStr) || ("N/A").equals(valueStr)) {
                    valueStr = null;
                } else {
                    if(this.evaluator.getDataField(fn).getDataType() == DataType.DOUBLE) {
                        try {
                            Double.parseDouble(valueStr);
                        } catch (Exception e) {
                            valueStr = null;
                        }
                    }
                }
                FieldValue value = EvaluatorUtil.prepare(this.evaluator, fn, (Object) valueStr);
                dataMap.put(fn, value);
            }

            @SuppressWarnings("unused")
            Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(dataMap);
            // for(Map.Entry<FieldName, Double> entry: regressionTerm.entrySet()) {
            // System.out.println(entry.getValue() + " " + map.get("diagnosis"));
            // }
        }
    }

    @SuppressWarnings("unused")
    @Test
    public void testEvalScore() throws IOException {
        List<String> lines = FileUtils.readLines(new File("src/test/resources/dttest/data/nnbinary.csv"));

        if(lines.size() <= 1) {
            return;
        }
        String[] headers = CommonUtils.split(lines.get(0), "|");
        // score with format <String, String>
        for(int i = 1; i < lines.size(); i++) {
            Map<String, String> map = new HashMap<String, String>();
            Map<String, Object> mapObj = new HashMap<String, Object>();

            String[] data = CommonUtils.split(lines.get(i), "|");;
            // System.out.println("data len is " + data.length);
            if(data.length != headers.length) {
                System.out.println("One invalid input data");
                break;
            }
            for(int j = 0; j < headers.length; j++) {
                map.put(headers[j], data[j]);
                mapObj.put(headers[j], data[j]);
            }
            double[] scores = iNNModel.compute(mapObj);
            ScoreObject scoreObject = scorer.score(map);
//            System.out.println("Eval score is: " + scoreObject.getMeanScore() / 1000 + "; bi score: " + scores[0]);
        }

    }

}
