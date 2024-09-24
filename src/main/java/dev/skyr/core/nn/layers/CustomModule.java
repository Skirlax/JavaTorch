package dev.skyr.core.nn.layers;

import dev.skyr.core.autograd.Tensor;

import java.util.HashMap;

public class CustomModule extends Module{
    @Override
    public HashMap<String, Tensor> parameters() {

        HashMap<String, Tensor> params = new HashMap<>();
        for (java.lang.reflect.Field field : this.getClass().getDeclaredFields()) {
            // check if the field is a subclass of Module
            if (Module.class.isAssignableFrom(field.getType())) {
                try {
                    // set the field to accessible
                    field.setAccessible(true);
                    // add the field to the parameters
                    HashMap<String, Tensor> fieldParams = ((Module) field.get(this)).parameters();
                    for (String key : fieldParams.keySet()) {
                        params.put(field.getName() + "." + key, fieldParams.get(key));
                    }
                } catch (IllegalAccessException e) {
                    e.printStackTrace();
                }
            }
        }
        return params;
    }
}
