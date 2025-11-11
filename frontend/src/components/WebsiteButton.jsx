const WebsiteButton = ({ children, isFullWidth, ...rest }) => {
  return (
    <button
      {...rest}
      className={`cursor-pointer bg-sky-400 text-white py-3 px-4 rounded-lg font-medium hover:opacity-90 transition transform hover:scale-[1.01] flex items-center justify-center ${isFullWidth && "w-full"}`}
    >
      {children}
    </button>
  )
}

export default WebsiteButton;
