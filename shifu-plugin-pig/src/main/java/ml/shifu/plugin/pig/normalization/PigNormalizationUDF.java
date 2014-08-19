package ml.shifu.plugin.pig.normalization;

import ml.shifu.core.di.builtin.executor.PMMLModelExecutor;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.pig.EvalFunc;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.xml.sax.InputSource;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;
import java.util.Set;

import javax.xml.transform.sax.SAXSource;

/**
 * NormalizeUDF class normalize the training data
 */
/*
public class NormalizeUDF extends AbstractTrainerUDF<Tuple> {

    private List<String> negTags;
    private List<String> posTags;
    private Expression weightExpr;
    private NormalizationService normalizationService;

    public NormalizeUDF(String source, String pathModelConfig, String pathColumnConfig) throws Exception {
        super(source, pathModelConfig, pathColumnConfig);

        log.debug("Initializing NormalizeUDF ... ");

        negTags = modelConfig.getNegTags();
        log.debug("\t Negative Tags: " + negTags);
        posTags = modelConfig.getPosTags();
        log.debug("\t Positive Tags: " + posTags);

        weightExpr = createExpression(modelConfig.getWeightColumnName());
        log.debug("NormalizeUDF Initialized");

        Injector injector = Guice.createInjector(new NormalizationModule(modelConfig.getNormalize().getNormalizer()));
        normalizationService = injector.getInstance(NormalizationService.class);
    }

    public Tuple exec(Tuple input) throws IOException {
        if (input == null || input.size() == 0) {
            return null;
        }

        int size = input.size();

        JexlContext jc = new MapContext();
        DecimalFormat df = new DecimalFormat("#.######");

        Tuple tuple = TupleFactory.getInstance().newTuple();

        String tag = input.get(tagColumnNum).toString();
        if (!(posTags.contains(tag) || negTags.contains(tag))) {
            log.warn("Invalid target column value - " + tag);
            return null;
        }

        boolean isNotSampled = DataSampler.isNotSampled(
                modelConfig.getPosTags(),
                modelConfig.getNegTags(),
                modelConfig.getNormalizeSampleRate(),
                modelConfig.isNormalizeSampleNegOnly(), tag);
        if (isNotSampled) {
            return null;
        }

        if (negTags.contains(tag)) {
            tuple.append(0);
        } else {
            tuple.append(1);
        }

        //Double cutoff = modelConfig.getNormalizeStdDevCutOff();
        /*
        for (int i = 0; i < size; i++) {
            ColumnConfig config = columnConfigList.get(i);
            if ( weightExpr != null ) {
                jc.set(config.getColumnName(), ((input.get(i) == null) ? "" : input.get(i).toString()));
            }
            
            if (config.isFinalSelect()) {
                String val = ((input.get(i) == null) ? "" : input.get(i).toString());

                Double z = Normalizer.normalize(modelConfig.getNormalize(), config, val);

                tuple.append(df.format(z));
            }
        }
        */
           /*
        List<Double> normalized = normalizationService.normalize(columnConfigList, input.getAll());

        if (normalized == null) {
            return null;
        }

        for (Double value : normalized) {
            tuple.append(df.format(value));
        }

        double weight = 1.0d;
        if (weightExpr != null) {
            Object result = weightExpr.evaluate(jc);
            if (result instanceof Integer) {
                weight = ((Integer) result).doubleValue();
            } else if (result instanceof Double) {
                weight = ((Double) result).doubleValue();
            }
        }
        tuple.append(weight);

        return tuple;
    }

    public Schema outputSchema(Schema input) {
        try {
            StringBuilder schemaStr = new StringBuilder();

            schemaStr.append("Normalized:Tuple(" + columnConfigList.get(tagColumnNum).getColumnName() + ":int");
            for (ColumnConfig config : columnConfigList) {
                if (config.isFinalSelect()) {
                    if (config.isNumerical()) {
                        schemaStr.append(", " + config.getColumnName() + ":float");
                    } else {
                        schemaStr.append(", " + config.getColumnName() + ":chararray");
                    }
                }
            }
            schemaStr.append(", weight:float)");

            return Utils.getSchemaFromString(schemaStr.toString());
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Create expressions for multi weight settings
     *
     * @param weightExprList
     * @return weight expression map
     */         /*
    protected Map<Expression, Double> createExpressionMap(List<WeightAmplifier> weightExprList) {
        Map<Expression, Double> ewMap = new HashMap<Expression, Double>();

        if (CollectionUtils.isNotEmpty(weightExprList)) {
            JexlEngine jexl = new JexlEngine();

            for (WeightAmplifier we : weightExprList) {
                ewMap.put(jexl.createExpression(we.getTargetExpression()), Double.valueOf(we.getTargetWeight()));
            }
        }

        return ewMap;
    }

    /**
     * Create the expression for weight setting
     *
     * @param weightAmplifier
     * @return expression for weight amplifier
     */  
/*
    private Expression createExpression(String weightAmplifier) {
        if (StringUtils.isNotBlank(weightAmplifier)) {
            JexlEngine jexl = new JexlEngine();
            return jexl.createExpression(weightAmplifier);
        }
        return null;
    }
    
}
*/