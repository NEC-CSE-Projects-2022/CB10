import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { ArrowDown, Satellite, Shield, Globe } from 'lucide-react';
import { Header } from '@/components/Header';
import { ImageUploader } from '@/components/ImageUploader';
import { ClassificationResults } from '@/components/ClassificationResults';
import { LandCoverClasses } from '@/components/LandCoverClasses';
import { StatsSection } from '@/components/StatsSection';
import { PaperAbstract } from '@/components/PaperAbstract';
import { classifyImages, type ValidationResult, type ClassificationResult } from '@/lib/satelliteValidator';
import heroImage from '@/assets/hero-satellite.jpg';

const Index = () => {
  const [validatedImages, setValidatedImages] = useState<Array<{ file: File; preview: string }>>([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [classificationResults, setClassificationResults] = useState<Array<{ result: ClassificationResult; file: File; preview: string }>>([]);

  const handleImagesValidated = useCallback(async (images: Array<{ file: File; preview: string }>) => {
    setValidatedImages(images);
    setIsProcessing(true);
    setClassificationResults([]);

    try {
      const results = await classifyImages(images);
      setClassificationResults(results);
    } catch (error) {
      console.error('Classification failed:', error);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const handleValidationFailed = useCallback((results: Array<{ file: File; result: ValidationResult }>) => {
    setValidatedImages([]);
    setClassificationResults([]);
    console.log('Validation failed for some images:', results);
  }, []);

  const resetClassifier = () => {
    setValidatedImages([]);
    setClassificationResults([]);
  };

  return (
    <div className="min-h-screen bg-background">
      <Header />

      {/* Hero Section */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden pt-16">
        {/* Background Image */}
        <div className="absolute inset-0 z-0">
          <img
            src={heroImage}
            alt="Earth from space"
            className="w-full h-full object-cover opacity-30"
          />
          <div className="absolute inset-0 bg-gradient-to-b from-background via-background/50 to-background" />
        </div>

        {/* Grid Pattern Overlay */}
        <div className="absolute inset-0 grid-pattern opacity-30 z-0" />

        {/* Content */}
        <div className="relative z-10 container mx-auto px-4 text-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-muted/50 border border-border mb-8"
            >
              <div className="w-2 h-2 rounded-full bg-success animate-pulse" />
              <span className="text-sm text-muted-foreground">Powered by YOLOv8 & EfficientNet</span>
            </motion.div>

            {/* Title */}
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="text-gradient-orbital">Satellite Image</span>
              <br />
              <span className="text-foreground">Land Cover Classification</span>
            </h1>

            {/* Description */}
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto mb-8">
              AI-powered forest monitoring and environmental analysis using deep learning 
              on Sentinel-2 satellite imagery. Classify 10 distinct land cover types with 97.5% accuracy.
            </p>

            {/* CTA */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12">
              <motion.a
                href="#classify"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="inline-flex items-center gap-2 px-8 py-4 bg-gradient-orbital text-white font-semibold rounded-xl orbital-glow"
              >
                <Satellite className="w-5 h-5" />
                Start Classification
              </motion.a>
              <motion.a
                href="#about"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                className="inline-flex items-center gap-2 px-8 py-4 bg-muted text-foreground font-semibold rounded-xl border border-border hover:border-primary/50 transition-colors"
              >
                Learn More
              </motion.a>
            </div>

            {/* Features */}
            <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-muted-foreground">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-primary" />
                <span>Satellite Image Validation</span>
              </div>
              <div className="flex items-center gap-2">
                <Globe className="w-4 h-4 text-primary" />
                <span>10 Land Cover Classes</span>
              </div>
              <div className="flex items-center gap-2">
                <Satellite className="w-4 h-4 text-primary" />
                <span>EuroSAT Dataset</span>
              </div>
            </div>
          </motion.div>

          {/* Scroll Indicator */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1, y: [0, 10, 0] }}
            transition={{ delay: 1, y: { duration: 2, repeat: Infinity } }}
            className="absolute bottom-8 left-1/2 -translate-x-1/2"
          >
            <ArrowDown className="w-6 h-6 text-muted-foreground" />
          </motion.div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-gradient-space">
        <div className="container mx-auto px-4">
          <StatsSection />
        </div>
      </section>

      {/* Paper Abstract Section */}
      <PaperAbstract />

      {/* About Section */}
      <section id="about" className="py-20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Multi-Zonal Forest Classification
            </h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Using YOLOv8 and EuroSAT dataset for scalable environmental monitoring. 
              Our model achieves 97.5% accuracy across 10 distinct land cover classes.
            </p>
          </motion.div>

          {/* Land Cover Classes */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="mb-16"
          >
            <h3 className="text-xl font-semibold mb-6 text-center">Supported Land Cover Classes</h3>
            <LandCoverClasses />
          </motion.div>

          {/* Research Info */}
          <div className="grid md:grid-cols-2 gap-8">
            <motion.div
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="stats-card rounded-xl p-6"
            >
              <h3 className="text-xl font-semibold mb-4">Research Objectives</h3>
              <ul className="space-y-3 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  Develop multi-zonal forest stand detection using YOLOv8 framework
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  Compare YOLOv8n with EfficientNet-B4 performance
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  Train and evaluate on 27,000 satellite images
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  Assess using accuracy, precision, recall, and F1-score
                </li>
              </ul>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="stats-card rounded-xl p-6"
            >
              <h3 className="text-xl font-semibold mb-4">Technical Stack</h3>
              <ul className="space-y-3 text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  <strong className="text-foreground">Dataset:</strong> EuroSAT (Sentinel-2 satellite imagery)
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  <strong className="text-foreground">Model:</strong> YOLOv8-nano / EfficientNet-B4
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  <strong className="text-foreground">Input:</strong> RGB spectral bands (64×64 patches)
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-primary">•</span>
                  <strong className="text-foreground">Accuracy:</strong> 97.5% (YOLOv8-nano)
                </li>
              </ul>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Classification Section */}
      <section id="classify" className="py-20 bg-gradient-space">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="text-center mb-12"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Classify Your Satellite Image
            </h2>
            <p className="text-muted-foreground max-w-xl mx-auto">
              Upload a satellite or aerial image to classify its land cover type. 
              The system validates that only satellite imagery is processed.
            </p>
          </motion.div>

          <div className="max-w-4xl mx-auto">
            {classificationResults.length === 0 ? (
              <ImageUploader
                onImagesValidated={handleImagesValidated}
                onValidationFailed={handleValidationFailed}
                isProcessing={isProcessing}
              />
            ) : (
              <div>
                <ClassificationResults
                  data={classificationResults}
                />
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  transition={{ delay: 0.5 }}
                  className="text-center mt-8"
                >
                  <button
                    onClick={resetClassifier}
                    className="px-6 py-3 bg-muted hover:bg-muted/80 rounded-xl font-medium transition-colors"
                  >
                    Classify More Images
                  </button>
                </motion.div>
              </div>
            )}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 border-t border-border">
        <div className="container mx-auto px-4 text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Satellite className="w-5 h-5 text-primary" />
            <span className="font-semibold">EuroSAT Land Cover Classifier</span>
          </div>
          <p className="text-sm text-muted-foreground">
            Enhanced Multi-Zonal Forest Type Classification Using YOLOv8 and EuroSAT
          </p>
          <p className="text-xs text-muted-foreground mt-2">
            Built for scalable environmental monitoring
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
