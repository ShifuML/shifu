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
package ml.shifu.shifu.core.dvarsel;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

import ml.shifu.guagua.master.MasterComputable;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created on 11/24/2014.
 */
public class VarSelMaster implements MasterComputable<VarSelMasterResult, VarSelWorkerResult> {

    private static final Logger LOG = LoggerFactory.getLogger(VarSelMaster.class);

    private AtomicBoolean isInitialized = new AtomicBoolean(false);

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;

    private AbstractMasterConductor masterConductor;

    @Override
    public VarSelMasterResult compute(MasterContext<VarSelMasterResult, VarSelWorkerResult> context) {
        if(this.isInitialized.compareAndSet(false, true)) {
            init(context);
        }

        if(context.getWorkerResults() == null) {
            throw new IllegalArgumentException("worker's result are null.");
        }

        masterConductor.consumeWorkerResults(context.getWorkerResults());
        LOG.info("Get results from workers ... ");

        VarSelMasterResult masterResult = new VarSelMasterResult(masterConductor.getNextWorkingSet());
        if(masterConductor.isToStop()) {
            LOG.info("Variables are selected. Send halt to workers ... ");
            masterResult.setHalt(true);

            // save the best seed
            masterResult.setBestSeed(masterConductor.voteBestSeed());
        } else {
            LOG.info("Send next working set to slaves ... ");
        }

        return masterResult;
    }

    private void init(MasterContext<VarSelMasterResult, VarSelWorkerResult> context) {
        Properties props = context.getProps();

        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));

            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);

            String conductorClsName = props.getProperty(Constants.VAR_SEL_MASTER_CONDUCTOR);

            this.masterConductor = (AbstractMasterConductor) Class.forName(conductorClsName)
                    .getDeclaredConstructor(ModelConfig.class, List.class)
                    .newInstance(this.modelConfig, this.columnConfigList);

        } catch (IOException e) {
            throw new RuntimeException("Fail to load ModelConfig or List<ColumnConfig>", e);
        } catch (ClassNotFoundException e) {
            throw new RuntimeException("Invalid Master Conductor class", e);
        } catch (InstantiationException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        }
    }

}
