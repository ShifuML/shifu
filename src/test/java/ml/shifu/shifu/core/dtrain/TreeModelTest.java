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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ml.shifu.guagua.util.FileUtils;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang3.tuple.MutablePair;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class TreeModelTest {

    private TreeModel model;

    private IndependentTreeModel iTreeModel;

    private Random random;

    @BeforeClass
    public void setUp() throws IOException {
        String modelPath = "src/test/resources/dttest/model/model_cam.gbt";
        FileInputStream fi = null;
        try {
            fi = new FileInputStream(modelPath);
            iTreeModel = IndependentTreeModel.loadFromStream(fi, true);
        } finally {
            fi.close();
        }

        fi = null;
        try {
            fi = new FileInputStream(modelPath);
            model = TreeModel.loadFromStream(fi);
        } finally {
            fi.close();
        }
        random = new Random();
    }

    @Test
    public void featureImportancesTest() {
        Map<Integer, MutablePair<String, Double>> importances = model.getFeatureImportances();
        Assert.assertTrue(importances.size() > 1);
    }

    @Test
    public void testScoring() throws IOException {
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/camdttest/config/ColumnConfig.json", SourceType.LOCAL);

        List<String> lines = FileUtils.readLines(new File("src/test/resources/dttest/data/tmdata.csv"));

        if(lines.size() <= 1) {
            return;
        }

        String[] headers = lines.get(0).split("\\|");
        // score with format <String, String>
        for(int i = 1; i < lines.size(); i++) {
            Map<String, Object> map = new HashMap<String, Object>();
            String[] data = lines.get(i).split("\\|");
            for(int j = 0; j < headers.length; j++) {
                map.put(headers[j], data[j]);
            }
            double[] scores = iTreeModel.compute(map);
            System.out.println(scores[0]);
            Assert.assertTrue(scores[0] >= 0 && scores[0] <= 1);
        }

        // score with format <String, Double> for numerical values
        for(int i = 1; i < lines.size(); i++) {
            Map<String, Object> map = new HashMap<String, Object>();
            String[] data = lines.get(i).split("\\|");
            for(int j = 0; j < headers.length; j++) {
                ColumnConfig columnConfig = columnConfigList.get(j);
                if(columnConfig.isCategorical()) {
                    map.put(headers[j], data[j]);
                } else {
                    try {
                        map.put(headers[j], Double.parseDouble(data[j]));
                    } catch (Exception e) {
                        map.put(headers[j], null);
                    }
                }

            }
            double[] scores = iTreeModel.compute(map);
            Assert.assertTrue(scores[0] >= 0 && scores[0] <= 1);
        }

        // score with format <String, Double> for numerical values while add some missing values for numeric feature
        for(int i = 1; i < lines.size(); i++) {
            Map<String, Object> map = new HashMap<String, Object>();
            String[] data = lines.get(i).split("\\|");
            for(int j = 0; j < headers.length; j++) {
                ColumnConfig columnConfig = columnConfigList.get(j);
                if(columnConfig.isCategorical()) {
                    map.put(headers[j], data[j]);
                } else {
                    double rr = random.nextDouble();
                    if(rr > 0.9) {
                        // random mock non numeric values or null
                        map.put(headers[j], rr > 0.95 ? "abc" : null);
                    } else {
                        try {
                            map.put(headers[j], Double.parseDouble(data[j]));
                        } catch (Exception e) {
                            map.put(headers[j], null);
                        }
                    }
                }
            }
            double[] scores = iTreeModel.compute(map);
            Assert.assertTrue(scores[0] >= 0 && scores[0] <= 1);
        }

        // score with format <String, Double> for numerical values while add some missing values for categorical feature
        for(int i = 1; i < lines.size(); i++) {
            Map<String, Object> map = new HashMap<String, Object>();
            String[] data = lines.get(i).split("\\|");
            for(int j = 0; j < headers.length; j++) {
                ColumnConfig columnConfig = columnConfigList.get(j);
                if(columnConfig.isCategorical()) {
                    double rr = random.nextDouble();
                    if(rr > 0.9) {
                        map.put(headers[j], rr > 0.95 ? null : "");
                    } else {
                        map.put(headers[j], data[j]);
                    }
                } else {
                    try {
                        map.put(headers[j], Double.parseDouble(data[j]));
                    } catch (Exception e) {
                        map.put(headers[j], null);
                    }
                }
            }
            double[] scores = iTreeModel.compute(map);
            Assert.assertTrue(scores[0] >= 0 && scores[0] <= 1);
        }

    }

}
