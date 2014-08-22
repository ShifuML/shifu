package ml.shifu.plugin.pig.normalization;

import ml.shifu.core.di.builtin.transform.DefaultTransformationExecutor;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.pig.EvalFunc;
import org.dmg.pmml.DataField;
import org.dmg.pmml.DerivedField;
import org.dmg.pmml.Expression;
import org.dmg.pmml.FieldColumnPair;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.FieldRef;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MapValues;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.Model;
import org.dmg.pmml.NormContinuous;
import org.dmg.pmml.PMML;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.xml.sax.InputSource;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.xml.transform.sax.SAXSource;

/**
 * NormalizeUDF class normalize the training data
 */

public class PigNormalizeUDF extends EvalFunc<Tuple> {

    private PMML pmml;
    private Model model;
    private DefaultTransformationExecutor executor;
    
    public PigNormalizeUDF(String pathPMML, String modelName)  throws Exception {

          
        this.pmml = loadPMML(pathPMML);
        
        for(Model m : pmml.getModels()) {
            if(m.getModelName().equalsIgnoreCase(modelName)) {
                model = m;
            }
        }
        log.debug("NormalizeUDF Initialized");
        
        executor = new DefaultTransformationExecutor();

        
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

        int size = input.size();

        Tuple tuple = TupleFactory.getInstance().newTuple();
        Map<String, Object> rawDataMap = new HashMap<String, Object>();
        List<DataField> dataFields = pmml.getDataDictionary().getDataFields();
        
        Map<FieldUsageType, List<DerivedField>> fieldsMap = getDerivedFieldsByUsageType(pmml, model);

        List<DerivedField> activeFields = fieldsMap.get(FieldUsageType.ACTIVE);
        List<DerivedField> targetFields = fieldsMap.get(FieldUsageType.TARGET);

        for (int i = 0; i < size; i++) {
            rawDataMap.put(dataFields.get(i).getName().getValue(), input.get(i).toString());
        }
        List<Object> result = executor.transform(targetFields, rawDataMap);
        result.addAll(executor.transform(activeFields, rawDataMap));
        for(Object o : result) {
            tuple.append(o);
        }

        return tuple;
    }

    public static Map<FieldUsageType, List<DerivedField>> getDerivedFieldsByUsageType(PMML pmml, Model model) {

        Map<FieldUsageType, List<DerivedField>> result = new LinkedHashMap<FieldUsageType, List<DerivedField>>();

        List<DerivedField> activeFields = new ArrayList<DerivedField>();
        List<DerivedField> targetFields = new ArrayList<DerivedField>();

        for (DerivedField derivedField : model.getLocalTransformations().getDerivedFields()) {
            Expression expression = derivedField.getExpression();

            List<MiningField> miningFields = model.getMiningSchema().getMiningFields();
            Map<FieldName,MiningField> miningMap = new HashMap<FieldName,MiningField>();
            for(MiningField m : miningFields) {
                miningMap.put(m.getName(), m);
            }
            
            if (expression instanceof NormContinuous) {
                for (MiningField miningField : miningFields) {
                    if (miningField.getName().equals(((NormContinuous) expression).getField())) {

                        if (miningField.getUsageType() == FieldUsageType.ACTIVE) {
                            activeFields.add(derivedField);
                        } else if (miningField.getUsageType() == FieldUsageType.TARGET) {
                            targetFields.add(derivedField);
                        }

                    }
                }
            } else if (expression instanceof MapValues) {
                int active = 0;
                int target = 0;
                List<FieldColumnPair> pairs = ((MapValues) expression).getFieldColumnPairs();
                for (FieldColumnPair pair : pairs) {
                    MiningField miningField = miningMap.get(pair.getField());
                    if (miningField != null) {
                        if (miningField.getUsageType() == FieldUsageType.ACTIVE) {
                            active += 1;
                        } else if (miningField.getUsageType() == FieldUsageType.TARGET) {
                            target += 1;
                        }
                    }
                }
                if (active == pairs.size()) {
                    activeFields.add(derivedField);
                } else if (target == pairs.size()) {
                    targetFields.add(derivedField);
                }

            } else if (expression instanceof FieldRef) {
                MiningField miningField = miningMap.get(((FieldRef) expression).getField());
                if (miningField != null) {
                    if (miningField.getUsageType() == FieldUsageType.ACTIVE) {
                        activeFields.add(derivedField);
                    } else if (miningField.getUsageType() == FieldUsageType.TARGET) {
                        targetFields.add(derivedField);
                    }
                }
            }
        }

        result.put(FieldUsageType.ACTIVE, activeFields);
        result.put(FieldUsageType.TARGET, targetFields);

        return result;
    }
     
}
