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

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Map;

import ml.shifu.shifu.core.TreeModel;

import org.apache.commons.lang3.tuple.MutablePair;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class TreeModelTest {

    private TreeModel model;

    @BeforeClass
    public void setUp() throws IOException {
        String modelPath = "src/test/resources/dttest/model/model_cam.gbt";
        FileInputStream fi = new FileInputStream(modelPath);
        model = TreeModel.loadFromStream(fi);
    }

    @Test
    public void FeatureImportancesTest() {
        Map<Integer, MutablePair<String, Double>> importances = model.getFeatureImportances();
        assert (importances.size() > 1);
    }

}
