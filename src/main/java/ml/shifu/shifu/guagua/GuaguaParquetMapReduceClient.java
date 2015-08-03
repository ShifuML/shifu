/*
 * Copyright [2013-2014] PayPal Software Foundation
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
package ml.shifu.shifu.guagua;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import ml.shifu.guagua.mapreduce.GuaguaMapReduceClient;

import org.apache.hadoop.mapreduce.Job;
import org.apache.pig.impl.logicalLayer.FrontendException;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.logicalLayer.schema.Schema.FieldSchema;
import org.apache.pig.impl.util.Utils;
import org.apache.pig.parser.ParserException;

import parquet.hadoop.ParquetInputFormat;
import parquet.hadoop.metadata.GlobalMetaData;
import parquet.io.ParquetDecodingException;
import parquet.pig.PigMetaData;
import parquet.pig.PigSchemaConverter;
import parquet.pig.SchemaConversionException;
import parquet.schema.IncompatibleSchemaModificationException;
import parquet.schema.MessageType;

/**
 * {@link GuaguaParquetMapReduceClient} is to append parquet format to job configuration which can be read in later
 * mappers.
 */
public class GuaguaParquetMapReduceClient extends GuaguaMapReduceClient {

    private static final PigSchemaConverter pigSchemaConverter = new PigSchemaConverter(false);

    /**
     * Create Hadoop job according to arguments from main.
     */
    @Override
    public synchronized Job createJob(String[] args) throws IOException {
        Job job = super.createJob(args);

        // for parquet format job, we have to append parquet schema field. We can only set parquet.pig.schema here
        // because of 'Job' dependency. While the other two required list parameters are in TrainModelProcessor.
        @SuppressWarnings("rawtypes")
        final GlobalMetaData globalMetaData = new ParquetInputFormat().getGlobalMetaData(job);
        Schema schema = getPigSchemaFromMultipleFiles(globalMetaData.getSchema(), globalMetaData.getKeyValueMetaData());
        String schemaStr = pigSchemaToString(schema);
        job.getConfiguration().set("parquet.pig.schema", schemaStr);

        return job;
    }

    /**
     * @param pigSchema
     *            the pig schema to turn into a string representation
     * @return the sctring representation of the schema
     */
    static String pigSchemaToString(Schema pigSchema) {
        final String pigSchemaString = pigSchema.toString();
        return pigSchemaString.substring(1, pigSchemaString.length() - 1);
    }

    /**
     * @param fileSchema
     *            the parquet schema from the file
     * @param keyValueMetaData
     *            the extra meta data from the files
     * @return the pig schema according to the file
     */
    static Schema getPigSchemaFromMultipleFiles(MessageType fileSchema, Map<String, Set<String>> keyValueMetaData) {
        Set<String> pigSchemas = PigMetaData.getPigSchemas(keyValueMetaData);
        if(pigSchemas == null) {
            return pigSchemaConverter.convert(fileSchema);
        }
        Schema mergedPigSchema = null;
        for(String pigSchemaString: pigSchemas) {
            try {
                mergedPigSchema = union(mergedPigSchema, parsePigSchema(pigSchemaString));
            } catch (FrontendException e) {
                throw new ParquetDecodingException("can not merge " + pigSchemaString + " into " + mergedPigSchema, e);
            }
        }
        return mergedPigSchema;
    }

    /**
     * @param pigSchemaString
     *            the pig schema to parse
     * @return the parsed pig schema
     */
    public static Schema parsePigSchema(String pigSchemaString) {
        try {
            return pigSchemaString == null ? null : Utils.getSchemaFromString(pigSchemaString);
        } catch (ParserException e) {
            throw new SchemaConversionException("could not parse Pig schema: " + pigSchemaString, e);
        }
    }

    private static Schema union(Schema merged, Schema pigSchema) throws FrontendException {
        List<FieldSchema> fields = new ArrayList<Schema.FieldSchema>();
        if(merged == null) {
            return pigSchema;
        }
        // merging existing fields
        for(FieldSchema fieldSchema: merged.getFields()) {
            FieldSchema newFieldSchema = pigSchema.getField(fieldSchema.alias);
            if(newFieldSchema == null) {
                fields.add(fieldSchema);
            } else {
                fields.add(union(fieldSchema, newFieldSchema));
            }
        }
        // adding new fields
        for(FieldSchema newFieldSchema: pigSchema.getFields()) {
            FieldSchema oldFieldSchema = merged.getField(newFieldSchema.alias);
            if(oldFieldSchema == null) {
                fields.add(newFieldSchema);
            }
        }
        return new Schema(fields);
    }

    private static FieldSchema union(FieldSchema mergedFieldSchema, FieldSchema newFieldSchema) {
        if(!mergedFieldSchema.alias.equals(newFieldSchema.alias) || mergedFieldSchema.type != newFieldSchema.type) {
            throw new IncompatibleSchemaModificationException("Incompatible Pig schema change: " + mergedFieldSchema
                    + " can not accept");
        }
        try {
            return new FieldSchema(mergedFieldSchema.alias, union(mergedFieldSchema.schema, newFieldSchema.schema),
                    mergedFieldSchema.type);
        } catch (FrontendException e) {
            throw new SchemaConversionException(e);
        }
    }

}
