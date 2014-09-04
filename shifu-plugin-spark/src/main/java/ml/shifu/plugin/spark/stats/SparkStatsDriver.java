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
package ml.shifu.plugin.spark.stats;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.Map;

import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.UnivariateStatsService;
import ml.shifu.core.request.Binding;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import ml.shifu.core.util.RequestUtils;
import ml.shifu.plugin.spark.stats.interfaces.SparkStatsCalculator;
import ml.shifu.plugin.spark.utils.CombinedUtils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.dmg.pmml.Model;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;

import com.google.inject.Guice;
import com.google.inject.Injector;

/**
 * Class called by spark-submit script. Needs a main method.
 * Arguments:
 *  1. HDFS Uri
 *  2. HDFS path to Input file
 *  3. HDFS path to PMML XML
 *  4. HDFS path to Request json
 */

public class SparkStatsDriver {

    public static void main(String[] args) throws IOException, URISyntaxException {
        String hdfsUri= args[0];
        String pathHdfsInput= args[1];
        String pathHdfsPmml= args[2];
        String pathHdfsRequest= args[3];
        
        FileSystem hdfs = FileSystem.get(new URI(hdfsUri), new Configuration());

        PMML pmml = CombinedUtils.loadPMML(pathHdfsPmml, hdfs);

        Request req = JSONUtils.readValue(hdfs.open(new Path(pathHdfsRequest)),
                Request.class);

        Params params = req.getProcessor().getParams();

        // TODO: Convert pathHDFSTmp to full hdfs path
        String pathHDFSTmp = (String) params.get("pathHDFSTmp",
                "hdfs://ml/shifu/plugin/spark/tmp");
        
        String appName = (String) params.get("SparkAppName", "spark-stats");
        SparkConf conf = new SparkConf().setAppName(appName);
        conf.set("spark.serializer",
                "org.apache.spark.serializer.KryoSerializer");
        JavaSparkContext jsc = new JavaSparkContext(conf);


        SimpleModule module = new SimpleModule();
        Binding statsCalculatorBinding = RequestUtils.getUniqueBinding(req, SparkStatsCalculator.class.getCanonicalName(), true);
        Params bindingParams= statsCalculatorBinding.getParams();
        module.set(statsCalculatorBinding);
        Injector injector = Guice.createInjector(module);

        // using Guice dependency injection
        SparkStatsService statsService= injector.getInstance(SparkStatsService.class);
        // SparkStatsCalculator sparkCalculator= new BinomialStatsCalculator();
        // create RDD
        JavaRDD<String> data= jsc.textFile(pathHdfsInput);
        
        ModelStats modelStats= statsService.calculate(jsc, data, pmml, bindingParams);
        // store univariateStats in pmml and save in pathPMML
        Model model = PMMLUtils.getModelByName(pmml, (String) bindingParams.get("modelName"));
        model.setModelStats(modelStats);
        
        // save PMML to HDFS tmp
        CombinedUtils.savePMML(pmml, pathHdfsPmml, hdfs);
        System.out.println("Exiting Driver, saved PMML");
    }

}
