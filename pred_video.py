def pred_video_and_write_heatmap(video_path, output_path, img_size=(225,225)):
  self.model.eval()
  trans = transforms.Compose([transforms.ToPILImage(),
                              transforms.Resize(img_size[:2]),
                              transforms.ToTensor()
  ])
  cap = cv2.VideoCapture(video_path)
  print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), 50, (frame_width, frame_height))
  with torch.no_grad() as tng:
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
          oh, ow = frame.shape[:2]
          if oh > ow:
            new_img = np.zeros((oh, oh, 3), np.uint8)
            cl = ((oh-ow)//2)
            new_img[:,cl:cl+ow] = frame
            clx = cl
            cly = 0
          else:
            new_img = np.zeros((ow, ow, 3), np.uint8)
            cl = ((ow-oh)//2)
            new_img[cl:cl+oh,:] = frame
            clx = 0
            cly = cl
          img = trans(new_img)
          img = torch.unsqueeze(img, 0)
          img = img.to(device)
          preds = model(img)
          preds = preds.cpu().numpy()[0]
          coor_x = []
          coor_y = []
          for i,pred in enumerate(preds[:7]):
            cx = np.argmax(pred)%pred.shape[0]
            cy = np.argmax(pred)//pred.shape[0]
            ovx = preds[i+7][cy,cx]*15
            ovy = preds[i+14][cy,cx]*15
            coor_x.append(int((cx*15+ovx)*max([ow,oh])/img_size[1])-clx)
            coor_y.append(int((cy*15+ovy)*max([ow,oh])/img_size[0])-cly)
          # print(preds)
          preds = np.vstack([coor_x, coor_y]).T
          cv2.polylines(frame, [preds], True, (0,0,255), 2)
          # for pred in preds:
          #   cv2.circle(frame,(pred[0],pred[1]), int(2*ow/img_size[1]), (0,0,255), -1)
          
          out.write(frame)
        else:
          break

    cap.release()
    out.release()