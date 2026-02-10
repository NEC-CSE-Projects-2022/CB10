import { motion } from 'framer-motion';
import { useState } from 'react';
import { EUROSAT_CLASSES } from '@/lib/satelliteValidator';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';

export function LandCoverClasses() {
  const [selectedClass, setSelectedClass] = useState<string | null>(null);
  const [isDialogOpen, setIsDialogOpen] = useState(false);

  const folderMap: Record<string, string> = {
    annual_crop: 'AnnualCrop',
    forest: 'Forest',
    herbaceous_vegetation: 'HerbaceousVegetation',
    highway: 'Highway',
    industrial: 'Industrial',
    pasture: 'Pasture',
    permanent_crop: 'PermanentCrop',
    residential: 'Residential',
    river: 'River',
    sea_lake: 'SeaLake',
  };

  const classImages: Record<string, string[]> = {
    annual_crop: ['AnnualCrop_53.jpg', 'AnnualCrop_946.jpg', 'AnnualCrop_1383.jpg', 'AnnualCrop_2044.jpg', 'AnnualCrop_2998.jpg'],
    forest: ['Forest_275.jpg', 'Forest_421.jpg', 'Forest_1768.jpg', 'Forest_1776.jpg', 'Forest_2793.jpg'],
    herbaceous_vegetation: ['HerbaceousVegetation_653.jpg', 'HerbaceousVegetation_1077.jpg', 'HerbaceousVegetation_1339.jpg', 'HerbaceousVegetation_1806.jpg', 'HerbaceousVegetation_2753.jpg'],
    highway: ['Highway_238.jpg', 'Highway_701.jpg', 'Highway_1165.jpg', 'Highway_1709.jpg', 'Highway_2131.jpg'],
    industrial: ['Industrial_49.jpg', 'Industrial_272.jpg', 'Industrial_2043.jpg', 'Industrial_2240.jpg', 'Industrial_2398.jpg'],
    pasture: ['Pasture_221.jpg', 'Pasture_310.jpg', 'Pasture_1182.jpg', 'Pasture_1226.jpg', 'Pasture_1995.jpg'],
    permanent_crop: ['PermanentCrop_40.jpg', 'PermanentCrop_151.jpg', 'PermanentCrop_1150.jpg', 'PermanentCrop_1697.jpg', 'PermanentCrop_2090.jpg'],
    residential: ['Residential_308.jpg', 'Residential_330.jpg', 'Residential_380.jpg', 'Residential_824.jpg', 'Residential_884.jpg'],
    river: ['River_492.jpg', 'River_1595.jpg', 'River_1920.jpg', 'River_2171.jpg', 'River_2179.jpg'],
    sea_lake: ['SeaLake_110.jpg', 'SeaLake_645.jpg', 'SeaLake_1455.jpg', 'SeaLake_2654.jpg', 'SeaLake_2981.jpg'],
  };

  const handleClassClick = (classId: string) => {
    setSelectedClass(classId);
    setIsDialogOpen(true);
  };

  return (
    <>
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        {EUROSAT_CLASSES.map((cls, index) => (
          <motion.div
            key={cls.id}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
            viewport={{ once: true }}
            className="stats-card rounded-lg p-3 group hover:border-primary/50 transition-colors cursor-pointer"
            onClick={() => handleClassClick(cls.id)}
            title="Click to view 5 sample images from EuroSAT dataset"
          >
            <div className="flex items-center gap-2 mb-1">
              <div
                className="w-3 h-3 rounded-full flex-shrink-0 ring-2 ring-offset-2 ring-offset-card group-hover:ring-primary/30 transition-all"
                style={{ backgroundColor: cls.color }}
              />
              <span className="text-sm font-medium truncate">{cls.name}</span>
            </div>
            <p className="text-xs text-muted-foreground line-clamp-2">
              {cls.description}
            </p>
          </motion.div>
        ))}
      </div>

      <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>
              {selectedClass ? EUROSAT_CLASSES.find(c => c.id === selectedClass)?.name : ''} Sample Images
            </DialogTitle>
          </DialogHeader>
          {selectedClass && (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {classImages[selectedClass].map((image, index) => (
                <div key={index} className="space-y-2">
                  <img
                    src={`/eurosat/${folderMap[selectedClass]}/${image}`}
                    alt={`${selectedClass} sample ${index + 1}`}
                    className="w-full h-32 object-cover rounded"
                  />
                  <Button asChild className="w-full">
                    <a href={`/eurosat/${folderMap[selectedClass]}/${image}`} download>
                      Download
                    </a>
                  </Button>
                </div>
              ))}
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
