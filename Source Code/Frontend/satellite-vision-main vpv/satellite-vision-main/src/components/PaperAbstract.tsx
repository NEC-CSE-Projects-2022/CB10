import { motion } from 'framer-motion';
import { FileText, Target, Database, Brain, CheckCircle } from 'lucide-react';

export function PaperAbstract() {
  const objectives = [
    "Develop a multi-zonal forest stand detection classification system using YOLOv8 framework",
    "Compare performance of YOLOv8n with EfficientNet-B4",
    "Train and evaluate models on 27,000 satellite images covering all land cover classes",
    "Assess model performance using accuracy, precision, recall, and F1-score"
  ];

  const keywords = [
    "Forest stand detection",
    "Remote sensing", 
    "YOLOv8",
    "EuroSAT",
    "Satellite imagery",
    "Land cover classification",
    "Deep learning",
    "Real-time monitoring"
  ];

  return (
    <section id="research" className="py-20 px-4">
      <div className="container mx-auto max-w-6xl">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary mb-4">
            <FileText className="w-4 h-4" />
            <span className="text-sm font-medium">Research Paper</span>
          </div>
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            Enhanced Multi-Zonal Forest Type Classification
          </h2>
          <p className="text-muted-foreground text-lg">
            Using YOLOv8 and EuroSAT for Scalable Environmental Monitoring
          </p>
        </motion.div>

        {/* Abstract Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="glass-card rounded-2xl p-8 mb-8"
        >
          <div className="flex items-center gap-3 mb-6">
            <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center">
              <Brain className="w-6 h-6 text-primary" />
            </div>
            <h3 className="text-2xl font-semibold">Abstract</h3>
          </div>
          <p className="text-muted-foreground leading-relaxed text-lg">
            Forests are a critical component of our world's health, but it is still difficult today to see them properly over large distances. This research proposes a real-world and intelligent solution using satellite images and Deep Learning to detect forest stands with greater accuracy. With the hope of faster and more scalable forest analysis than is achievable using today's traditional field surveys, we set out to create an automatic identification system for land cover classes from the EuroSAT dataset. The main goal of this study was to train and compare the YOLOv8-nano model to see how well it detects forests and other land cover types. The model was tested on <span className="text-foreground font-semibold">27,000 images</span> from <span className="text-foreground font-semibold">10 different classes</span>. Our evaluation employed accuracy, precision, recall, and F1-score, and the YOLOv8-nano model produced the best accuracy of <span className="text-primary font-bold">97.5%</span>. These results show the high potential of the model for real-time forest monitoring and environmental decision-making.
          </p>
        </motion.div>

        {/* Objectives & Dataset */}
        <div className="grid md:grid-cols-2 gap-8 mb-8">
          {/* Research Objectives */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="glass-card rounded-2xl p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-lg bg-success/20 flex items-center justify-center">
                <Target className="w-5 h-5 text-success" />
              </div>
              <h3 className="text-xl font-semibold">Research Objectives</h3>
            </div>
            <ul className="space-y-4">
              {objectives.map((objective, index) => (
                <motion.li
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  whileInView={{ opacity: 1, x: 0 }}
                  viewport={{ once: true }}
                  transition={{ delay: 0.3 + index * 0.1 }}
                  className="flex items-start gap-3"
                >
                  <CheckCircle className="w-5 h-5 text-success flex-shrink-0 mt-0.5" />
                  <span className="text-muted-foreground">{objective}</span>
                </motion.li>
              ))}
            </ul>
          </motion.div>

          {/* Dataset Info */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="glass-card rounded-2xl p-6"
          >
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
                <Database className="w-5 h-5 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">EuroSAT Dataset</h3>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center p-3 rounded-lg bg-muted/50">
                <span className="text-muted-foreground">Total Images</span>
                <span className="font-mono font-bold text-primary">27,000</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-lg bg-muted/50">
                <span className="text-muted-foreground">Land Cover Classes</span>
                <span className="font-mono font-bold text-primary">10</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-lg bg-muted/50">
                <span className="text-muted-foreground">Image Size</span>
                <span className="font-mono font-bold text-primary">64×64 px</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-lg bg-muted/50">
                <span className="text-muted-foreground">Source Satellite</span>
                <span className="font-mono font-bold text-primary">Sentinel-2</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-lg bg-muted/50">
                <span className="text-muted-foreground">Model Accuracy</span>
                <span className="font-mono font-bold text-success">97.5%</span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Keywords */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="text-center"
        >
          <h4 className="text-sm font-medium text-muted-foreground mb-4">Keywords</h4>
          <div className="flex flex-wrap justify-center gap-2">
            {keywords.map((keyword, index) => (
              <motion.span
                key={keyword}
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                viewport={{ once: true }}
                transition={{ delay: 0.4 + index * 0.05 }}
                className="px-4 py-2 rounded-full bg-muted text-muted-foreground text-sm hover:bg-primary/20 hover:text-primary transition-colors cursor-default"
              >
                {keyword}
              </motion.span>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
