package ml.shifu.shifu.core.dvarsel;
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

import ml.shifu.guagua.master.BasicMasterInterceptor;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.NNConstants;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;

/**
 * Created on 11/24/2014.
 */
public class VarSelOutput extends BasicMasterInterceptor<VarSelMasterResult, VarSelWorkerResult> {

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Column Config list read from HDFS
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * A flag: whether params initialized.
     */
    private AtomicBoolean isInit = new AtomicBoolean(false);

    @Override
    public void preApplication(MasterContext<VarSelMasterResult, VarSelWorkerResult> context) {

        final Properties props = context.getProps();

        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(NNConstants.NN_MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(NNConstants.SHIFU_NN_MODEL_CONFIG),sourceType);
            this.columnConfigList = CommonUtils.loadColumnConfigList(props.getProperty(NNConstants.SHIFU_NN_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }

    @Override
    public void postApplication(MasterContext<VarSelMasterResult, VarSelWorkerResult> context) {
        VarSelMasterResult varSelMasterResult = context.getMasterResult();

        List<Integer> results = varSelMasterResult.getColumnIdList();

        Path out = new Path(context.getProps().getProperty(Constants.VAR_SEL_COLUMN_IDS_OUPUT));

        writeColumnIdsIntoHDFS(out, results);
    }

    private void writeColumnIdsIntoHDFS(Path path, List<Integer> columnIds){

        FSDataOutputStream fos = null;
        try {
            fos = FileSystem.get(new Configuration()).create(path);
            final PrintWriter pw = new PrintWriter(fos);
            pw.println(Integer.toString(columnIds.size()) + "|" + columnIds.toString());

        } catch (IOException e) {
        	e.printStackTrace();
        	
        } finally {
            IOUtils.closeStream(fos);
        }
    }



}
