/*
 * Copyright [2013-2014] eBay Software Foundation
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
import java.io.PrintWriter;
import java.util.Arrays;

import ml.shifu.guagua.master.BasicMasterInterceptor;
import ml.shifu.guagua.master.MasterContext;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link LogisticRegressionOutput} is used to write the final model output to file system.
 */
public class LogisticRegressionOutput extends
        BasicMasterInterceptor<LogisticRegressionParams, LogisticRegressionParams> {

    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionOutput.class);

    /**
     * Get output file setting and write final sum value to HDFS file.
     */
    @Override
    public void postApplication(final MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        LOG.info("Starts to write final value to file.");
        Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
        LOG.info("Writing results to {}", out.toString());
        PrintWriter pw = null;
        try {
            FSDataOutputStream fos = FileSystem.get(new Configuration()).create(out);
            pw = new PrintWriter(fos);
            pw.println(Arrays.toString(context.getMasterResult().getParameters()));
            pw.flush();
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(pw);
        }
    }
}
