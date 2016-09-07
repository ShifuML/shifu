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
package ml.shifu.shifu.core.dtrain;

import java.io.IOException;
import java.util.Properties;

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.hadoop.GuaguaMRUnitDriver;
import ml.shifu.guagua.unit.GuaguaUnitDriver;
import ml.shifu.shifu.core.dtrain.dt.DTMaster;
import ml.shifu.shifu.core.dtrain.dt.DTMasterParams;
import ml.shifu.shifu.core.dtrain.dt.DTWorker;
import ml.shifu.shifu.core.dtrain.dt.DTWorkerParams;

//import org.testng.annotations.Test;

/**
 * RF & GBT d-train logic change
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class DTTest {

    // @Test
    public void testDtApp() throws IOException {
        Properties props = new Properties();
        props.setProperty(GuaguaConstants.MASTER_COMPUTABLE_CLASS, DTMaster.class.getName());
        props.setProperty(GuaguaConstants.WORKER_COMPUTABLE_CLASS, DTWorker.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_ITERATION_COUNT, "20");
        props.setProperty(GuaguaConstants.GUAGUA_MASTER_RESULT_CLASS, DTMasterParams.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_WORKER_RESULT_CLASS, DTWorkerParams.class.getName());
        props.setProperty(CommonConstants.MODELSET_SOURCE_TYPE, "LOCAL");
        props.setProperty(CommonConstants.SHIFU_MODEL_CONFIG, getClass().getResource("/dttest/config/ModelConfig.json")
                .toString());
        props.setProperty(CommonConstants.SHIFU_COLUMN_CONFIG,
                getClass().getResource("/dttest/config/ColumnConfig.json").toString());

        props.setProperty(GuaguaConstants.GUAGUA_INPUT_DIR, getClass().getResource("/dttest/data/").toString());

        GuaguaUnitDriver<DTMasterParams, DTWorkerParams> driver = new GuaguaMRUnitDriver<DTMasterParams, DTWorkerParams>(
                props);

        driver.run();
    }

    // @Test
    public void testCamDtApp() throws IOException {
        Properties props = new Properties();
        props.setProperty(GuaguaConstants.MASTER_COMPUTABLE_CLASS, DTMaster.class.getName());
        props.setProperty(GuaguaConstants.WORKER_COMPUTABLE_CLASS, DTWorker.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_ITERATION_COUNT, "20");
        props.setProperty(GuaguaConstants.GUAGUA_MASTER_RESULT_CLASS, DTMasterParams.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_WORKER_RESULT_CLASS, DTWorkerParams.class.getName());
        props.setProperty(CommonConstants.MODELSET_SOURCE_TYPE, "LOCAL");
        props.setProperty(CommonConstants.SHIFU_MODEL_CONFIG,
                getClass().getResource("/camdttest/config/ModelConfig.json").toString());
        props.setProperty(CommonConstants.SHIFU_COLUMN_CONFIG,
                getClass().getResource("/camdttest/config/ColumnConfig.json").toString());

        props.setProperty(GuaguaConstants.GUAGUA_INPUT_DIR, getClass().getResource("/camdttest/data/").toString());

        GuaguaUnitDriver<DTMasterParams, DTWorkerParams> driver = new GuaguaMRUnitDriver<DTMasterParams, DTWorkerParams>(
                props);

        driver.run();
    }

}
