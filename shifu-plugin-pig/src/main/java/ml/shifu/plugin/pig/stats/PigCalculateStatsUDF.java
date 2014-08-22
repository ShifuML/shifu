package ml.shifu.plugin.pig.stats;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.transform.sax.SAXSource;
import javax.xml.transform.stream.StreamResult;

import ml.shifu.core.container.RawValueObject;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.UnivariateStatsService;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.Params;

import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.pig.EvalFunc;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.dmg.pmml.DataField;
import org.dmg.pmml.Model;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.NeuralNetwork;
import org.dmg.pmml.PMML;
import org.dmg.pmml.UnivariateStats;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.xml.sax.InputSource;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.inject.AbstractModule;
import com.google.inject.Guice;
import com.google.inject.Injector;

/**
 * PigCalculateStatsUDF class is calculate the stats for each column
 * <p/>
 * Input: (columnNum, {(value, tag, weight), (value, tag, weight)...})
 */

public class PigCalculateStatsUDF extends EvalFunc<Tuple> {

    private Double valueThreshold = 1e6;

    //private PigStatsService pigStatsService;
    
    private UnivariateStatsService univariateStatsService;

    private Params params;

    private PMML pmml;

    private String spi;

    public PigCalculateStatsUDF(String request) throws Exception {


        ObjectMapper jsonMapper = new ObjectMapper();
        Request req = jsonMapper.readValue(request, Request.class);
        /*
        AbstractModule pigStatsInjector = new PigSimpleUnivariateStatsInjector();

        if (!req.getBindings().get(0).getSpi()
                .equalsIgnoreCase("UnivariateStatsCalculator")) {
            return;
        }

        if (req.getBindings().get(0).getImpl()
                .equalsIgnoreCase("PigBinomialUnivariateStatsCalculator")) {
            pigStatsInjector = new PigBinomialUnivariateStatsInjector();
        }

        
*/
        SimpleModule module = new SimpleModule();
        module.set(req.getBindings().get(0));
        Injector injector = Guice.createInjector(module);
        
        univariateStatsService = injector.getInstance(UnivariateStatsService.class);
        
        //pigStatsService = injector.getInstance(PigStatsService.class);

        params = req.getBindings().get(0).getParams();
              
        spi = req.getBindings().get(0).getSpi();

        pmml = loadPMML((String) req.getProcessor().getParams().get("pathPMML"));

    }

    public static PMML loadPMML(String path) throws Exception {

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path pmmlFilePath = new Path(path);
        FSDataInputStream in = fs.open(pmmlFilePath);

        try {
            InputSource source = new InputSource(in);
            SAXSource transformedSource = ImportFilter.apply(source);
            return JAXBUtil.unmarshalPMML(transformedSource);

        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
    }

    public Tuple exec(Tuple input) throws IOException {
        if (input == null || input.size() == 0) {
            return null;
        }

        TupleFactory tupleFactory = TupleFactory.getInstance();

        Integer columnNum = (Integer) input.get(0);
        DataBag bag = (DataBag) input.get(1);

        List<RawValueObject> rvoList = new ArrayList<RawValueObject>();

        log.debug("****** The element count in bag is : " + bag.size());
        List<Object> tagData = new ArrayList<Object>();
        for (Tuple t : bag) {
            RawValueObject rvo = new RawValueObject();
            rvo.setValue(t.get(0).toString());
            rvo.setTag(t.get(1).toString());
            rvo.setWeight(Double.valueOf(t.get(2).toString()));
            rvoList.add(rvo);
            tagData.add(t.get(2).toString());
        }

        DataField field = pmml.getDataDictionary().getDataFields()
                .get(columnNum);

        UnivariateStats stats = new UnivariateStats();

        
        
        params.put("tags", tagData);
  
        if (spi.equalsIgnoreCase("UnivariateStatsCalculator")) {
            stats = univariateStatsService.getUnivariateStats(field, rvoList, params);
            stats.setField(field.getName());

            for (Model model : pmml.getModels()) {
                if (model.getModelName().equalsIgnoreCase(
                        (String) params.get("modelName"))) {
                    model.getModelStats().getUnivariateStats().add(stats);
                }
            }
        }

        ModelStats modelStats = new ModelStats();
        modelStats.getUnivariateStats().add(stats);

        PMML pmml2 = new PMML();
        pmml2.getModels().add(new NeuralNetwork()); // Any model type will do
                                                    // here
        pmml2.getModels().get(0).setModelStats(modelStats);

        Tuple tuple = tupleFactory.newTuple();
        tuple.append(columnNum);
        tuple.append(savePMML(pmml2));

        return tuple;

    }

    public static String savePMML(PMML pmml) {
        OutputStream os = null;

        try {
            os = new ByteArrayOutputStream();
            StreamResult result = new StreamResult(os);
            JAXBUtil.marshalPMML(pmml, result);
        } catch (Exception e) {
            // log.error(e.toString());
        } finally {
            IOUtils.closeQuietly(os);
        }

       String ret = os.toString().replaceAll("\n", "").replaceAll("\\s+", " ");
        return ret;
    }
}
