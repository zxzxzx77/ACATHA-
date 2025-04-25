class AdaptiveContextIntegration(nn.Module):
    def __init__(self, feature_dim=128):  # Reduced feature dimension
        super(AdaptiveContextIntegration, self).__init__()
        
        
        self.confidence_predictor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim//4, kernel_size=3, padding=1),  # Reduced from feature_dim//2
            nn.BatchNorm2d(feature_dim//4),
            nn.ReLU(),
            nn.Conv2d(feature_dim//4, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        
        self.context_processor = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
            # Removed the third dilated convolution to save memory
        )
        
    def forward(self, x):
        # Process in two steps to save memory
        confidence = self.confidence_predictor(x)
        context_features = self.context_processor(x)
        
        
        integrated_features = x * (1 - confidence) + context_features * confidence
        
        
        del context_features
        torch.cuda.empty_cache()
        
        return integrated_features, confidence