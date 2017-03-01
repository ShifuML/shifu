/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.varsel;

import java.io.IOException;
import java.util.Properties;

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.hadoop.GuaguaMRUnitDriver;
import ml.shifu.guagua.unit.GuaguaUnitDriver;
import ml.shifu.shifu.core.dvarsel.VarSelMaster;
import ml.shifu.shifu.core.dvarsel.VarSelMasterResult;
import ml.shifu.shifu.core.dvarsel.VarSelWorker;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;

public class VotedVarSelTest {

    // @Test
    public void testLrApp() throws IOException {
        Properties props = new Properties();

        props.setProperty(GuaguaConstants.MASTER_COMPUTABLE_CLASS, VarSelWorker.class.getName());
        props.setProperty(GuaguaConstants.WORKER_COMPUTABLE_CLASS, VarSelMaster.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_ITERATION_COUNT, "20");
        props.setProperty(GuaguaConstants.GUAGUA_MASTER_RESULT_CLASS, VarSelMasterResult.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_WORKER_RESULT_CLASS, VarSelWorkerResult.class.getName());

        props.setProperty(GuaguaConstants.GUAGUA_INPUT_DIR,
                getClass().getResource("/example/cancer-judgement/DataStore/DataSet1/part-00").toString());

        GuaguaUnitDriver<VarSelMasterResult, VarSelWorkerResult> driver = new GuaguaMRUnitDriver<VarSelMasterResult, VarSelWorkerResult>(
                props);

        driver.run();

    }

}
