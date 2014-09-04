/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.plugin.spark.norm;

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.conf.Configuration;
import org.dmg.pmml.*;

import ml.shifu.core.di.builtin.transform.DefaultTransformationExecutor;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.UnivariateStatsService;
import ml.shifu.core.request.Binding;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.Params;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.core.util.RequestUtils;
import ml.shifu.plugin.spark.utils.CombinedUtils;
import ml.shifu.plugin.spark.utils.HDFSFileUtils;

import java.net.URI;
import java.util.List;

import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import com.google.inject.Guice;
import com.google.inject.Injector;
/**
 * This is the class that is called by spark-submit. Hence, it requires a main method.
 * The three arguments required for this class are the paths to the PMML XML model file and the request JSON file.
 * In the main method, 
 * 	1. The paths and other parameters are unpacked from the request object
 * 	2. SparkConf and JavaSparkContext objects are created 
 * 	3. Broadcast variables are populated
 * 	4. JavaRDD is created from the input file
 * 	5. Map operation is done on the RDD using the Normalize class
 * 	6. resulting JavaRDD is stored as a text file in the input
 * 
 */

public class SparkNormalizer {
    public static void main(String[] args) throws Exception {
        // argument 1: HDFS root Uri
        // argument 2: HDFS path to input data file
        // argument 3: HDFS path to request.json
        // argument 4: HDFS path to PMML model.xml
        String hdfsUri = args[0];
        String pathHDFSInputData = args[1];
        String pathHDFSPmml = args[2];
        String pathHDFSReq = args[3];

        FileSystem fs = FileSystem.get(new URI(hdfsUri), new Configuration());

        PMML pmml = CombinedUtils.loadPMML(pathHDFSPmml, fs);

        Request req = JSONUtils.readValue(fs.open(new Path(pathHDFSReq)),
                Request.class);

        Params params = req.getProcessor().getParams();

        // TODO: Convert pathHDFSTmp to full hdfs path

        String pathHDFSTmp = (String) params.get("pathHDFSTmp",
                "ml/shifu/plugin/spark/tmp");
        
        HDFSFileUtils hdfsUtils= new HDFSFileUtils(new URI(hdfsUri));
        pathHDFSTmp= hdfsUtils.relativeToFullHDFSPath(pathHDFSTmp);
        String precision = (String) params.get("precision", "3");
        String delimiter = (String) params.get("delimiter", ",");
        String appName = (String) params.get("SparkAppName", "spark-norm");

        // TODO: add interface TransformExecutor, use DI
        /*
         * SimpleModule module = new SimpleModule(); Binding
         * dataDictionaryCreatorBinding = RequestUtils.getUniqueBinding(req,
         * "TransformationExecutor"); module.set(dataDictionaryCreatorBinding);
         * Injector injector = Guice.createInjector(module); TransformExecutor
         * executor = injector.getInstance(UnivariateStatsService.class);
         */
        DefaultTransformationExecutor executor = new DefaultTransformationExecutor();

        // SparkConf conf= new
        // SparkConf().setAppName(appName).setMaster(yarnMode);
        SparkConf conf = new SparkConf().setAppName(appName);
        conf.set("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer");
        conf.set("spark.kyro.Registrator", "ml.shifu.norm.MyRegistrator");
        JavaSparkContext jsc = new JavaSparkContext(conf);
        List<DerivedField> activeFields = CombinedUtils.getActiveFields(pmml,
                params);
        List<DerivedField> targetFields = CombinedUtils.getTargetFields(pmml,
                params);

        Broadcast<BroadcastVariables> bVar = jsc
                .broadcast(new BroadcastVariables(executor, pmml, pmml
                        .getDataDictionary().getDataFields(), activeFields,
                        targetFields, precision, delimiter));

        JavaRDD<String> raw = jsc.textFile(pathHDFSInputData);
        JavaRDD<String> normalized = raw.map(new Normalize(bVar));

        // create the output in a pathHDFSTmp/output directory. The files will
        // be concatenated into a single file.
        normalized.saveAsTextFile(pathHDFSTmp + "/" + "output");

    }

}
