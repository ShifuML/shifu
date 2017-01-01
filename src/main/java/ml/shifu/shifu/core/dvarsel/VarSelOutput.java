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

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

import ml.shifu.guagua.master.BasicMasterInterceptor;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.hadoop.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created on 11/24/2014.
 */
public class VarSelOutput extends BasicMasterInterceptor<VarSelMasterResult, VarSelWorkerResult> {

    private static final Logger LOG = LoggerFactory.getLogger(VarSelOutput.class);

    /**
     * Model Config read from HDFS
     */
    @SuppressWarnings("unused")
    private ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    @SuppressWarnings("unused")
    private List<ColumnConfig> columnConfigList;

    /**
     * A flag: whether params initialized.
     */
    @SuppressWarnings("unused")
    private AtomicBoolean isInit = new AtomicBoolean(false);

    @Override
    public void preApplication(MasterContext<VarSelMasterResult, VarSelWorkerResult> context) {

        final Properties props = context.getProps();

        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public void postApplication(MasterContext<VarSelMasterResult, VarSelWorkerResult> context) {
        VarSelMasterResult varSelMasterResult = context.getMasterResult();

        LOG.info("Results is - {}", varSelMasterResult.getBestSeed());

        String out = context.getProps().getProperty(Constants.VAR_SEL_COLUMN_IDS_OUPUT);

        writeColumnIdsIntoHDFS(out, varSelMasterResult.getBestSeed().getColumnIdList());
    }

    private void writeColumnIdsIntoHDFS(String path, List<Integer> columnIds) {
        BufferedWriter bw = null;
        try {
            bw = ShifuFileUtils.getWriter(path, SourceType.HDFS);
            bw.write(String.format("%s|%s", Integer.toString(columnIds.size()), columnIds.toString()));
            bw.newLine();
            bw.flush();
        } catch (IOException e) {
            e.printStackTrace();
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(bw);
        }
    }

}
