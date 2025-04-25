class HierarchicalFeatureExtractor(nn.Module):
    def __init__(self, in_channels=3, feature_dim=128):  
        super(HierarchicalFeatureExtractor, self).__init__()
        
        
        self.full_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),   32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  64
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        
        self.half_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        
        self.quarter_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),   32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        
        self.eighth_branch = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),   64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),  64
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        
        self.fusion = nn.Sequential(
            nn.Conv2d(32*4, feature_dim, kernel_size=1, stride=1),  
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        
    def forward(self, x):
        
        full_features = self.full_branch(x)
        half_features = self.half_branch(x)
        quarter_features = self.quarter_branch(x)
        eighth.features = self.eighth_branch(x)
        
       
        h, w = full_features.size(2), full_features.size(3)
        half_up = nn.functional.interpolate(half_features, size=(h, w), mode='bilinear', align_corners=True)
        quarter_up = nn.functional.interpolate(quarter_features, size=(h, w), mode='bilinear', align_corners=True)
        eighth_up = nn.functional.interpolate(eighth_features, size=(h, w), mode='bilinear', align_corners=True)
        
        
        concat_features = torch.cat([full_features, half_up, quarter_up, eighth_up], dim=1)
        
        
        fused_features = self.fusion(concat_features)
        
        
        del full_features, half_features. quarter_features, eighth_features, half_up, quarter_up, eighth_up, concat_features
        torch.cuda.empty_cache()
        
        return fused_features